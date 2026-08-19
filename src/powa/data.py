"""CIFAR input pipeline for O-VGG training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def dataset_spec(
    dataset: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    name = dataset.lower()
    if name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10
    if name == "cifar100":
        return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761), 100
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def build_train_transform(dataset: str) -> transforms.Compose:
    """Return the configured augmentation pipeline with explicit defaults."""

    mean, standard_deviation, _ = dataset_spec(dataset)
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="constant"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
            transforms.RandomErasing(
                p=0.4,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
    )


def build_test_transform(dataset: str) -> transforms.Compose:
    mean, standard_deviation, _ = dataset_spec(dataset)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )


def _dataset_class(dataset: str):
    dataset_spec(dataset)
    return datasets.CIFAR10 if dataset.lower() == "cifar10" else datasets.CIFAR100


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker = torch.utils.data.get_worker_info()
    if worker is None:
        return
    worker_seed = worker.seed % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_train_and_test_loaders(
    dataset: str,
    data_directory: str | Path,
    *,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """Load the official 50,000-train/10,000-test CIFAR split.

    The test loader supports fixed-schedule reporting curves and the final
    rounded-deployment measurement. It must not be used for checkpoint
    selection or hyperparameter tuning.
    """

    _, _, num_classes = dataset_spec(dataset)
    dataset_class = _dataset_class(dataset)
    train_set = dataset_class(
        root=str(data_directory),
        train=True,
        download=True,
        transform=build_train_transform(dataset),
    )
    test_set = dataset_class(
        root=str(data_directory),
        train=False,
        download=True,
        transform=build_test_transform(dataset),
    )
    if len(train_set) != 50_000 or len(test_set) != 10_000:
        raise ValueError("expected the official 50,000-train/10,000-test CIFAR split")
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader, num_classes


def get_test_loader(
    dataset: str,
    data_directory: str | Path,
    *,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, int]:
    """Load only the official test split for checkpoint evaluation."""

    _, _, num_classes = dataset_spec(dataset)
    dataset_class = _dataset_class(dataset)
    test_set = dataset_class(
        root=str(data_directory),
        train=False,
        download=True,
        transform=build_test_transform(dataset),
    )
    if len(test_set) != 10_000:
        raise ValueError("expected the official 10,000-example CIFAR test split")
    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, num_classes
