import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss_LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

def make_loss(name):
    if name == "mse":
        return nn.MSELoss()

    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif name == "bce":
        return nn.BCEWithLogitsLoss()

    elif name == "cross_entropy_smooth":
        return CELoss_LabelSmoothing()

    raise ValueError(f"Unknown loss: {name}")