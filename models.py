import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def param_count(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total

def make_model(config, dataset_metadata):
    model_config = config['model']
    if model_config['name'] == 'mlp':
        return MLP(
            input_dims=dataset_metadata.num_features, 
            hidden_dims=model_config['hidden_dims'],
            output_dims=dataset_metadata.num_classes, 
            num_layers=model_config['num_layers'], 
        )

    elif model_config['name'] == 'min-max-plus':
        return MinMaxPlus(
            input_dims=dataset_metadata.num_features, 
            hidden_dims=model_config['hidden_dims'],
            output_dims=dataset_metadata.num_classes, 
            num_layers=model_config['num_layers'], 
            temperature=model_config['temperature'], 
            cheap=model_config['cheap'], 
        )

    elif model_config['name'] == 'ViT':
        return ViT(
            image_size=dataset_metadata.input_shape[1:], 
            patch_size=model_config['patch_size'], 
            num_classes=dataset_metadata.num_classes, 
            dim=model_config['embed_dim'], 
            depth=model_config['num_layers'], 
            heads=4, 
            mlp_dim=model_config['hidden_dims'], 
            channels=dataset_metadata.input_shape[0], 
        )

    elif model_config['name'] == 'MorphViT':
        return MorphViT(
            image_size=dataset_metadata.input_shape[1:],  
            patch_size=model_config['patch_size'], 
            num_classes=dataset_metadata.num_classes, 
            dim=model_config['embed_dim'], 
            depth=model_config['num_layers'], 
            heads=4, 
            mlp_dim=model_config['hidden_dims'],
            channels=dataset_metadata.input_shape[0],  
            temperature=model_config['temperature'], 
            cheap=model_config['cheap'], 
        )

    raise ValueError(f"Unknown model name {model_config['name']}")

class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, num_layers=1):
        super().__init__()

        self.input_dims = input_dims

        prev = input_dims
        self.layers = []
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(prev, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
            ))
            prev = hidden_dims
        self.layers = nn.ModuleList(self.layers)

        self.out = nn.Sequential(
            nn.Linear(prev, hidden_dims), 
            nn.ReLU(), 
            nn.Linear(hidden_dims, output_dims), 
        )

    def forward(self, x):
        x = x.view(-1, self.input_dims)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

class MinMaxPlus(nn.Module):
    def __init__(
            self, 
            input_dims, 
            hidden_dims, 
            output_dims, 
            num_layers=1,
            temperature=1.0, 
            cheap=False, 
        ):
        super().__init__()

        self.input_dims = input_dims

        prev = input_dims
        self.layers = []
        for _ in range(num_layers-1):
            self.layers.append(nn.Sequential(
                LinAct(2*prev),
                MorphologicalLayer(2*prev, hidden_dims, bias=False, method='max', temperature=temperature, cheap=cheap), 
                LinAct(hidden_dims), 
                MorphologicalLayer(hidden_dims, hidden_dims, bias=False, method='min', temperature=temperature, cheap=cheap), 
            ))
            prev = hidden_dims
        self.layers = nn.ModuleList(self.layers)

        self.out = nn.Sequential(
            LinAct(2*prev),
            MorphologicalLayer(2*prev, hidden_dims, bias=False, method='max', temperature=temperature, cheap=cheap), 
            LinAct(hidden_dims), 
            MorphologicalLayer(hidden_dims, output_dims, bias=False, method='min', temperature=temperature, cheap=cheap), 
        )

    def forward(self, x):
        x = x.view(-1, self.input_dims)
        for layer in self.layers:
            x = torch.cat([x, -x], dim=-1)
            x = layer(x)
        x = torch.cat([x, -x], dim=-1)
        return self.out(x)

def maxplus_layer(x, weight, bias=None):
    x = x.unsqueeze(1)
    x = x + weight
    if bias is not None:
        x = torch.cat([x, bias.view(1,-1,1).repeat(x.size(0),1,1)], dim=2)
    x, arg = torch.max(x, dim=2)
    return x

def smaxplus_layer(x, weight, bias=None, T=1.0):
    x = x.unsqueeze(1)
    x = x + weight
    if bias is not None:
        x = torch.cat([x, bias.view(1,-1,1).repeat(x.size(0),1,1)], dim=2)
    x = T*torch.logsumexp(x/T, dim=2)
    return x

def smaxplus_layer_cheap(x, weight, bias=None, T=1.0):
    if bias is not None:
        # Interpret bias as one additional constant expert.
        x = torch.cat([
            x,
            torch.zeros(
                x.size(0), 1,
                device=x.device,
                dtype=x.dtype,
            ),
        ], dim=1)

        weight = torch.cat([
            weight,
            bias[:, None],
        ], dim=1)

    x_scaled = x / T
    w_scaled = weight / T

    # Per-sample and per-output stabilizers.
    x_shift = x_scaled.max(dim=1, keepdim=True).values       # [B, 1]
    w_shift = w_scaled.max(dim=1, keepdim=True).values      # [O, 1]

    x_exp = torch.exp(x_scaled - x_shift)                   # [B, I]
    w_exp = torch.exp(w_scaled - w_shift)                   # [O, I]

    z = F.linear(x_exp, w_exp)                              # [B, O]

    return T * (
        torch.log(z)
        + x_shift
        + w_shift.transpose(0, 1)
    )

class MaxPlusLayer(nn.Module):
    def __init__(
            self, 
            input_dims, 
            output_dims, 
            bias=True, 
            mean=0, 
            std=1, 
            temperature=1.0, 
            cheap=False, 
        ):
        super().__init__()
        self.weight = nn.Parameter(torch.normal(
            mean=mean, std=std, size=(output_dims, input_dims)
        ))
        self.c = nn.Parameter(torch.randn(output_dims))
        self.bias = nn.Parameter(torch.randn(output_dims)) if bias else None

        self.input = None
        self.output = None

        self.temperature = temperature
        self.cheap = cheap

    def forward(self, x):
        self.input = x
        if self.temperature != 0.0:
            if self.cheap: 
                # x = torch.exp(x / self.temperature)
                # max_w = torch.exp(self.weight / self.temperature)
                # max_b = torch.exp(self.bias / self.temperature) if self.bias is not None else None
                # self.out = self.temperature * torch.log(F.linear(x, max_w, max_b))
                self.out = smaxplus_layer_cheap(x, self.weight, self.bias, self.temperature)
                assert self.out.isfinite().all()
            else:
                self.out = smaxplus_layer(x, self.weight, self.bias, self.temperature)
        else:
            self.out = maxplus_layer(x, self.weight, self.bias)
        if self.training:
            self.out.retain_grad()
        return (self.out + self.c)

class MorphologicalLayer(nn.Module):
    def __init__(
            self, 
            input_dims, 
            output_dims, 
            bias=True, 
            mean=0, 
            std=1.0, 
            method='max',
            temperature=1.0, 
            cheap=False,
        ):
        super().__init__()

        self.method = method
        self.maxplus = MaxPlusLayer(input_dims, output_dims, bias, mean, std, temperature, cheap)

    def forward(self, x):
        if self.method == 'max':
            return self.maxplus(x)
        elif self.method == 'min':
            return -self.maxplus(-x)
        else:
            raise ValueError('Unsupported morphological method.')

class LinAct(nn.Module):
    def __init__(self, dims):
        super(LinAct, self).__init__()
        self.dims = dims
        self.a = nn.Parameter(torch.randn(dims))

    def forward(self, x):
        x = x * self.a.view(1, -1)
        return x


######################
# ViTs
######################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



class MorphFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., inverse=False, temperature=1.0, cheap=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            LinAct(dim), 
            MorphologicalLayer(dim, hidden_dim, method = 'min' if inverse else 'max', temperature=temperature, cheap=cheap), 
            nn.Dropout(dropout),
            LinAct(hidden_dim), 
            MorphologicalLayer(hidden_dim, dim, method = 'max' if inverse else 'min', temperature=temperature, cheap=cheap),
            nn.Dropout(dropout), 
            LinAct(dim), 
        )

    def forward(self, x):
        B, N, _ = x.size()
        x = x.view(B*N, -1)
        return self.net(x).view(B,N,-1)
        
class MorphTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., temperature=1.0, cheap=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                MorphFeedForward(dim, mlp_dim, dropout = dropout, temperature=temperature, cheap=cheap), 
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class MorphViT(nn.Module):
    def __init__(
            self, 
            *, 
            image_size, 
            patch_size, 
            num_classes, 
            dim, depth, 
            heads, 
            mlp_dim, 
            pool = 'cls', 
            channels = 3, 
            dim_head = 64, 
            dropout = 0., 
            emb_dropout = 0.,
            temperature=1.0, 
            cheap=False, 
        ):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = MorphTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, temperature=temperature, cheap=cheap)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)