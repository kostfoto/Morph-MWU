import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, loader, metric, config):
    model.eval()
    metric.reset()

    for x, y in loader:
        x = x.to(config["device"])
        y = y.to(config["device"])

        outputs = model(x)
        metric.update(outputs, y)

    return metric.compute()