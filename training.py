import torch
import copy
import tqdm
import matplotlib.pyplot as plt

from evaluation import evaluate
from models import MaxPlusLayer

def normalize(model):
    for module in model.modules():
        if isinstance(module, MaxPlusLayer):
            if module.temperature != 0.0:
                if module.bias is not None:
                    weight_bias = torch.cat(
                        [module.weight, module.bias[:, None]],
                        dim=1,
                    )

                    norm = module.temperature * torch.logsumexp(
                        weight_bias / module.temperature,
                        dim=1,
                        keepdim=True,
                    )
                    module.weight.data -= norm
                    module.bias.data -= norm.squeeze(1)
                else: 
                    module.weight.data -= module.temperature * torch.logsumexp(module.weight.data / module.temperature, dim=1, keepdim=True)
            else:
                if module.bias is not None: 
                    weight_bias = torch.cat([module.weight.data, module.bias.data.view(-1,1)], dim=1)
                    norm = torch.max(weight_bias, dim=1, keepdim=True)[0]
                    module.weight.data -= norm
                    module.bias.data -= norm.squeeze(1)
                else:
                    module.weight.data -= torch.max(module.weight.data, dim=1, keepdim=True)[0]  

def train(
    model,
    optimizer,
    criterion,
    metric, 
    train_loader,
    val_loader,
    config,
    print_output=True, 
):
    device = config["device"]

    train_loss = []
    train_metric = []
    val_metric = []

    best_metric = None
    checkpoint_path = (
        config["output"]["runs_root"]
        / config["output"]["run_name"]
        / "checkpoints"
        / "best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if config['use_mwu']:
        normalize(model)

    for epoch in range(config["epochs"]):
        model.train()
        metric.reset()

        total_loss = 0.0
        total = 0

        for i, (x, labels) in enumerate(tqdm.tqdm(train_loader)):
            # Batch norm guard
            if x.size(0) == 1: 
                continue
            x = x.to(device)
            labels = labels.to(device)

            outputs = model(x)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            if config['use_mwu']:
                for module in model.modules():
                    if isinstance(module, MaxPlusLayer):
                        with torch.no_grad(): 
                            assert module.out is not None and module.input is not None
                            assert module.out.grad is not None
                            delta = module.out.grad.detach()
                            inp = module.input.detach()
                            if module.bias is not None:
                                inp = torch.cat([module.input, 
                                                torch.ones(module.input.size(0), 1).to(module.input.device), 
                                                ], dim=-1)
                            out = module.out.detach()
                            delta -= delta.mean(dim=0, keepdim=True)
                            delta = -delta
                            corr = delta.unsqueeze(2) * (inp.unsqueeze(1) - out.unsqueeze(2))
                            corr = corr.sum(dim=0)
                            if module.bias is not None:
                                module.weight.grad = -corr[:, :-1]
                                module.bias.grad = -corr[:, -1]
                            else:
                                module.weight.grad = -corr

            optimizer.step()

            if config['use_mwu']:
                normalize(model)              

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total += batch_size

            metric.update(outputs, labels)

        epoch_loss = total_loss / total
        train_loss.append(epoch_loss)

        train_metric.append(metric.compute())

        val_metric.append(
            evaluate(model, val_loader, metric, config)
        )

        if print_output:
            print("Epoch: {}, Train_loss: {:.4f}, Train metric: {:.4f}, Val metric: {:.4f}".format(
                epoch+1, epoch_loss, train_metric[-1], val_metric[-1]
            ))

        if metric.is_better(best_metric):
            best_metric = copy.deepcopy(metric)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metric": val_metric,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config["device"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "train_loss": train_loss,
        "train_metric": train_metric,
        "val_metric": val_metric,
    }