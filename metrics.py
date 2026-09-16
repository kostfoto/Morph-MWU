import torch

def make_metric(name):
    if name == "accuracy":
        return Accuracy()

    if name == "balanced_accuracy":
        return BalancedAccuracy()

    if name == "mse":
        return MSE()

    raise ValueError(f"Unknown metric: {name}")

class Accuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        predictions = outputs.argmax(dim=1)
        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()

    def compute(self):
        return self.correct / self.total

    def is_better(self, metric2):
        if metric2 is None:
            return True
        return self.compute() > metric2.compute()

class MSE:
    def __init__(self):
        self.reset()

    def reset(self):
        self.squared_error = 0.0
        self.total = 0

    def update(self, outputs, targets):
        self.squared_error += ((outputs - targets) ** 2).sum().item()
        self.total += targets.numel()

    def compute(self):
        return self.squared_error / self.total

    def is_better(self, metric2):
        if metric2 is None:
            return True
        return self.compute() < metric2.compute()

class BalancedAccuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_per_class = None
        self.total_per_class = None

    def update(self, outputs, targets):
        predictions = outputs.argmax(dim=1)

        num_classes = outputs.size(1)

        if self.correct_per_class is None:
            self.correct_per_class = torch.zeros(
                num_classes,
                dtype=torch.long,
            )
            self.total_per_class = torch.zeros(
                num_classes,
                dtype=torch.long,
            )

        targets_cpu = targets.detach().cpu()
        predictions_cpu = predictions.detach().cpu()

        self.total_per_class += torch.bincount(
            targets_cpu,
            minlength=num_classes,
        )

        correct_targets = targets_cpu[
            predictions_cpu == targets_cpu
        ]

        self.correct_per_class += torch.bincount(
            correct_targets,
            minlength=num_classes,
        )

    def compute(self):
        present = self.total_per_class > 0

        recalls = (
            self.correct_per_class[present].float()
            / self.total_per_class[present].float()
        )

        return recalls.mean().item()

    def is_better(self, metric2):
        if metric2 is None:
            return True

        return self.compute() > metric2.compute()