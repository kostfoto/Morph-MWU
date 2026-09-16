import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)

from pmlb import fetch_data, classification_dataset_names
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from reproducibility import seed_worker


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    input_shape: Tuple[int, ...]
    num_classes: int

    @property
    def num_features(self):
        n = 1

        for dim in self.input_shape:
            n *= dim

        return n

def make_dataloaders(config):
    train_dataset, val_dataset, test_dataset, metadata = make_datasets(
        config
    )

    train_loader = make_loader(
        train_dataset,
        config,
        shuffle=True,
    )

    val_loader = make_loader(
        val_dataset,
        config,
        shuffle=False,
    )

    test_loader = make_loader(
        test_dataset,
        config,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        metadata,
    )


def make_loader(dataset, config, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["dataset"]["num_workers"],
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(
            config["seed"]
        ),
        pin_memory=True,
    )


def make_datasets(config):
    dataset_config = config["dataset"]
    name = dataset_config["name"].lower()

    if name == "mnist":
        return make_mnist_datasets(config)

    if name == "fashion-mnist":
        return make_fashion_mnist_datasets(config)

    if name == "cifar-10":
        return make_cifar10_datasets(config)

    if name == "pmlb":
        return make_pmlb_datasets(config)

    raise ValueError(
        f"Unknown dataset: {name}"
    )


# ---------------------------------------------------------------------------
# Image datasets
# ---------------------------------------------------------------------------


def make_mnist_datasets(config):
    dataset_config = config["dataset"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,),
        ),
    ])

    full_train_dataset = torchvision.datasets.MNIST(
        root=dataset_config["root"],
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root=dataset_config["root"],
        train=False,
        transform=transform,
        download=True,
    )

    train_dataset, val_dataset = split_torch_dataset(
        full_train_dataset,
        val_split=dataset_config["val_split"],
        seed=config["seed"],
    )

    metadata = infer_dataset_metadata(
        name="mnist",
        dataset=train_dataset,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        metadata,
    )


def make_fashion_mnist_datasets(config):
    dataset_config = config["dataset"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.2860,),
            (0.3530,),
        ),
    ])

    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_config["root"],
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_config["root"],
        train=False,
        transform=transform,
        download=True,
    )

    train_dataset, val_dataset = split_torch_dataset(
        full_train_dataset,
        val_split=dataset_config["val_split"],
        seed=config["seed"],
    )

    metadata = infer_dataset_metadata(
        name="fashion-mnist",
        dataset=train_dataset,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        metadata,
    )

def make_cifar10_datasets(config):
    dataset_config = config["dataset"]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    # Same underlying CIFAR-10 training examples,
    # but different transforms.
    full_train_aug = torchvision.datasets.CIFAR10(
        root=dataset_config["root"],
        train=True,
        transform=transform_train,
        download=True,
    )

    full_train_clean = torchvision.datasets.CIFAR10(
        root=dataset_config["root"],
        train=True,
        transform=transform_test,
        download=True,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_config["root"],
        train=False,
        transform=transform_test,
        download=True,
    )

    val_size = int(
        dataset_config["val_split"] * len(full_train_aug)
    )
    train_size = len(full_train_aug) - val_size

    generator = torch.Generator().manual_seed(config["seed"])
    perm = torch.randperm(
        len(full_train_aug),
        generator=generator,
    )

    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    train_dataset = torch.utils.data.Subset(
        full_train_aug,
        train_indices,
    )

    val_dataset = torch.utils.data.Subset(
        full_train_clean,
        val_indices,
    )

    metadata = infer_dataset_metadata(
        name="cifar-10",
        dataset=train_dataset,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        metadata,
    )

def split_torch_dataset(
    dataset,
    val_split,
    seed,
):
    val_size = int(
        val_split * len(dataset)
    )

    train_size = (
        len(dataset) - val_size
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    return random_split(
        dataset,
        [
            train_size,
            val_size,
        ],
        generator=generator,
    )


# ---------------------------------------------------------------------------
# PMLB
# ---------------------------------------------------------------------------

def make_pmlb_datasets(config):
    dataset_config = config["dataset"]
    dataset_name = dataset_config["pmlb_name"]

    if dataset_name not in classification_dataset_names:
        raise ValueError(
            f"Unknown PMLB classification dataset: {dataset_name!r}.\n"
            f"Available datasets include:\n"
            f"{', '.join(classification_dataset_names)}"
        )

    X, y = fetch_data(
        dataset_name,
        return_X_y=True,
        local_cache_dir=dataset_config.get(
            "root",
            "./data",
        ),
    )

    X = np.asarray(X)
    y = np.asarray(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    num_classes = len(
        label_encoder.classes_
    )

    val_split = dataset_config["val_split"]

    test_split = dataset_config.get(
        "test_split",
        0.2,
    )

    if not 0 < val_split < 1:
        raise ValueError(
            "val_split must be between 0 and 1"
        )

    if not 0 < test_split < 1:
        raise ValueError(
            "test_split must be between 0 and 1"
        )

    if val_split + test_split >= 1:
        raise ValueError(
            "val_split + test_split must be less than 1"
        )

    seed = config["seed"]

    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_split,
            random_state=seed,
            stratify=y,
        )
    )

    relative_val_split = (
        val_split /
        (1.0 - test_split)
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_split,
            random_state=seed,
            stratify=y_train_val,
        )
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    train_dataset = make_tensor_dataset(
        X_train,
        y_train,
    )

    val_dataset = make_tensor_dataset(
        X_val,
        y_val,
    )

    test_dataset = make_tensor_dataset(
        X_test,
        y_test,
    )

    metadata = DatasetMetadata(
        name=dataset_name,
        input_shape=(X_train.shape[1],),
        num_classes=num_classes,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        metadata,
    )


def make_tensor_dataset(X, y):
    return TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y,
            dtype=torch.long,
        ),
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def infer_dataset_metadata(
    name,
    dataset,
):
    """
    Infer metadata directly from a PyTorch dataset.

    This avoids maintaining a separate metadata registry.
    """

    x, _ = dataset[0]

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    input_shape = tuple(
        x.shape
    )

    num_classes = infer_num_classes(
        dataset
    )

    return DatasetMetadata(
        name=name,
        input_shape=input_shape,
        num_classes=num_classes,
    )


def infer_num_classes(dataset):
    """
    Infer the number of classes without maintaining
    a dataset-name -> num_classes registry.
    """

    base_dataset = unwrap_dataset(
        dataset
    )

    # torchvision datasets generally expose `classes`.
    if hasattr(base_dataset, "classes"):
        return len(
            base_dataset.classes
        )

    # Some datasets expose targets but not classes.
    if hasattr(base_dataset, "targets"):
        targets = base_dataset.targets

        targets = torch.as_tensor(
            targets
        )

        return int(
            torch.unique(targets).numel()
        )

    # Generic fallback.
    labels = []

    for _, y in dataset:
        if torch.is_tensor(y):
            y = y.item()

        labels.append(y)

    return len(
        set(labels)
    )


def unwrap_dataset(dataset):
    """
    Follow wrappers such as torch.utils.data.Subset
    until the underlying dataset is reached.
    """

    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    return dataset