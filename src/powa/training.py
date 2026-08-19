"""Optimization utilities for O-VGG classification."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sost import SOSTConv2d, project_sost_wavelengths


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 400
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def smoothed_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    log_probabilities = F.log_softmax(logits, dim=1)
    negative_log_likelihood = -log_probabilities.gather(
        dim=1,
        index=target.unsqueeze(1),
    ).squeeze(1)
    uniform_loss = -log_probabilities.mean(dim=1)
    return (
        (1.0 - smoothing) * negative_log_likelihood + smoothing * uniform_loss
    ).mean()


def make_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-3,
) -> torch.optim.Adam:
    """Create Adam groups with explicit weight-decay exclusions."""

    wavelength_ids = {
        id(module.wavelength_u)
        for module in model.modules()
        if isinstance(module, SOSTConv2d)
    }
    batch_norm_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for parameter in module.parameters(recurse=False)
    }
    bias_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("bias")
    }
    no_decay_ids = wavelength_ids | batch_norm_ids | bias_ids
    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        destination = (
            no_decay_parameters if id(parameter) in no_decay_ids else decay_parameters
        )
        destination.append(parameter)
    return torch.optim.Adam(
        [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
                "group_name": "decay",
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
                "group_name": "no_decay",
            },
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
    )


def run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    label_smoothing: float = 0.0,
    gradient_clip_norm: float = 0.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = smoothed_cross_entropy(logits, labels, smoothing=label_smoothing)
            if training:
                loss.backward()
                if gradient_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                optimizer.step()
                project_sost_wavelengths(model)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("the data loader produced no examples")
    return total_loss / total_examples, total_correct / total_examples
