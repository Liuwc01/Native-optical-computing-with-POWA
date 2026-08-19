"""Training utilities for the complete optical style-transfer network."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torchvision
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ..codebook import WPUCodebook
from ..sost import (
    SOSTConv2d,
    project_sost_wavelengths,
    set_sost_deployment,
    validate_sost_codebook_buffers,
    validate_sost_state_dict_codebook,
)
from .config import StyleTrainingConfig
from .data import ImagePathDataset, content_training_transform
from .losses import StyleTransferLoss, gram_matrix, style_transfer_loss_from_grams
from .model import (
    POWAStyleTransferNetwork,
    finalize_style_network,
    style_parameter_statistics,
)
from .perceptual import VGG16FeatureExtractor


@dataclass(frozen=True)
class EpochMetrics:
    total: float
    content: float
    style: float
    regularization: float
    batches: int


def _model_codebook(module: nn.Module) -> WPUCodebook:
    for child in module.modules():
        if isinstance(child, SOSTConv2d):
            reference = WPUCodebook(
                child.codebook.detach().cpu(),
                child.wavelength_nm.detach().cpu(),
            )
            validate_sost_codebook_buffers(module, reference)
            return reference
    raise ValueError("model contains no SOSTConv2d layers to identify its codebook")


def _software_versions() -> dict[str, str | None]:
    return {
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
    }


def load_style_model_state(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    codebook: WPUCodebook,
) -> None:
    """Load a style model without allowing its codebook to change silently."""

    if "model" in checkpoint:
        raw_state = checkpoint["model"]
        fingerprint = checkpoint.get("codebook_fingerprint")
        if fingerprint is not None:
            if not isinstance(fingerprint, str):
                raise ValueError("checkpoint codebook fingerprint must be a string")
            if fingerprint != codebook.fingerprint:
                raise ValueError(
                    "the codebook does not match the checkpoint fingerprint"
                )
    else:
        # Legacy pure state_dict files have no wrapper metadata, but their
        # persisted SOST buffers still provide a complete value-level check.
        raw_state = checkpoint

    if not isinstance(raw_state, Mapping):
        raise ValueError("checkpoint model state must be a mapping")
    state_dict = cast(Mapping[str, torch.Tensor], raw_state)
    validate_sost_state_dict_codebook(state_dict, codebook)
    model.load_state_dict(state_dict, strict=True)
    validate_sost_codebook_buffers(model, codebook)


def create_content_loader(
    root: str | Path,
    config: StyleTrainingConfig | None = None,
    *,
    workers: int = 0,
    shuffle: bool = True,
) -> DataLoader[torch.Tensor]:
    settings = config or StyleTrainingConfig()
    dataset = ImagePathDataset(root, content_training_transform(settings))
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
    )


def create_style_optimizer(
    model: POWAStyleTransferNetwork,
    config: StyleTrainingConfig | None = None,
) -> AdamW:
    """Create AdamW from trainable parameters only."""

    settings = config or StyleTrainingConfig()
    if settings.optimizer != "AdamW":
        raise ValueError("the reported style-transfer optimizer is AdamW")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("the style-transfer model has no trainable parameters")
    return AdamW(
        trainable_parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )


def create_style_scheduler(
    optimizer: Optimizer,
    config: StyleTrainingConfig | None = None,
) -> StepLR:
    settings = config or StyleTrainingConfig()
    return StepLR(
        optimizer,
        step_size=settings.decay_every_epochs,
        gamma=settings.decay_factor,
    )


class StyleTransferTrainer:
    """Train all configured style banks with paper-aligned preprocessing."""

    def __init__(
        self,
        model: POWAStyleTransferNetwork,
        feature_extractor: VGG16FeatureExtractor,
        style_references: Sequence[torch.Tensor],
        *,
        config: StyleTrainingConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or StyleTrainingConfig()
        if len(style_references) != self.config.styles:
            raise ValueError(
                f"expected {self.config.styles} style references, got {len(style_references)}"
            )
        if len(model.style_banks) != self.config.styles:
            raise ValueError("model and configuration use different style counts")
        if tuple(feature_extractor.layers) != tuple(self.config.style_vgg_layers):
            raise ValueError("feature extractor and configuration use different VGG layers")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.codebook = _model_codebook(self.model)
        # The default configuration fixes the encoder. With 1,966,080 bank
        # parameters and 394,848 decoder parameters, the banks contain 83.2757%
        # (approximately 84%) of optimized parameters. Fixed encoder parameters
        # are excluded from this denominator.
        self.model.set_encoder_trainable(not self.config.freeze_encoder)
        self.feature_extractor = feature_extractor.to(self.device).eval()
        self.feature_extractor.requires_grad_(False)
        self.style_references = [
            reference.unsqueeze(0).to(self.device) if reference.ndim == 3 else reference.to(self.device)
            for reference in style_references
        ]
        self.optimizer = create_style_optimizer(self.model, self.config)
        self.scheduler = create_style_scheduler(self.optimizer, self.config)
        self.epoch = 0
        self._style_cursor = 0
        set_sost_deployment(self.model, False)
        self._style_grams = self._cache_style_grams()

    @torch.no_grad()
    def _cache_style_grams(self) -> list[tuple[torch.Tensor, ...]]:
        return [
            tuple(
                gram_matrix(feature).detach()
                for feature in self.feature_extractor(reference)
            )
            for reference in self.style_references
        ]

    def _next_style_ids(self, batch_size: int) -> list[int]:
        identifiers = [
            (self._style_cursor + offset) % self.config.styles
            for offset in range(batch_size)
        ]
        self._style_cursor = (self._style_cursor + batch_size) % self.config.styles
        return identifiers

    def _batch_loss(
        self,
        output: torch.Tensor,
        content: torch.Tensor,
        style_ids: Sequence[int],
    ) -> StyleTransferLoss:
        output_features = self.feature_extractor(output)
        with torch.no_grad():
            content_features = self.feature_extractor(content)
        try:
            content_layer = self.config.style_vgg_layers.index(
                self.config.content_vgg_layer
            )
        except ValueError as error:
            raise ValueError("content VGG layer must also be extracted") from error

        sample_losses = []
        for index, style_id in enumerate(style_ids):
            sample_losses.append(
                style_transfer_loss_from_grams(
                    [feature[index : index + 1] for feature in output_features],
                    [feature[index : index + 1] for feature in content_features],
                    self._style_grams[style_id],
                    output[index : index + 1],
                    content_layer=content_layer,
                    style_layer_weights=self.config.style_layer_weights,
                    content_weight=self.config.content_weight,
                    style_weight=self.config.style_weight,
                    regularization_weight=self.config.regularization_weight,
                )
            )

        def mean(component: str) -> torch.Tensor:
            return torch.stack([getattr(loss, component) for loss in sample_losses]).mean()

        return StyleTransferLoss(
            content=mean("content"),
            style=mean("style"),
            regularization=mean("regularization"),
            total=mean("total"),
        )

    def train_epoch(self, content_batches: Iterable[torch.Tensor]) -> EpochMetrics:
        self.model.train()
        set_sost_deployment(self.model, False)
        totals = {"total": 0.0, "content": 0.0, "style": 0.0, "regularization": 0.0}
        batches = 0
        for content in content_batches:
            if isinstance(content, (tuple, list)):
                content = content[0]
            content = content.to(self.device)
            style_ids = self._next_style_ids(content.shape[0])
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(content, style_ids)
            losses = self._batch_loss(output, content, style_ids)
            losses.total.backward()
            self.optimizer.step()
            project_sost_wavelengths(self.model)
            for name in totals:
                totals[name] += float(getattr(losses, name).detach())
            batches += 1
        if not batches:
            raise ValueError("content loader produced no batches")
        self.scheduler.step()
        self.epoch += 1
        return EpochMetrics(
            total=totals["total"] / batches,
            content=totals["content"] / batches,
            style=totals["style"] / batches,
            regularization=totals["regularization"] / batches,
            batches=batches,
        )

    def fit(
        self,
        content_batches: Iterable[torch.Tensor],
        *,
        epochs: int | None = None,
        checkpoint_directory: str | Path | None = None,
    ) -> list[EpochMetrics]:
        target_epochs = self.config.epochs if epochs is None else int(epochs)
        if target_epochs < 1:
            raise ValueError("epochs must be positive")
        history: list[EpochMetrics] = []
        directory = Path(checkpoint_directory) if checkpoint_directory else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        for _ in range(target_epochs):
            metrics = self.train_epoch(content_batches)
            history.append(metrics)
            if directory is not None:
                self.save_checkpoint(directory / f"epoch_{self.epoch:03d}.pt")
        finalize_style_network(self.model)
        if directory is not None:
            self.save_checkpoint(directory / "deployment.pt")
        return history

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        validate_sost_codebook_buffers(self.model, self.codebook)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "style_cursor": self._style_cursor,
                "config": asdict(self.config),
                "parameter_statistics": style_parameter_statistics(self.model).to_dict(),
                "codebook_fingerprint": self.codebook.fingerprint,
                "software_versions": _software_versions(),
            },
            target,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        load_style_model_state(self.model, checkpoint, self.codebook)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epoch = int(checkpoint["epoch"])
        self._style_cursor = int(checkpoint.get("style_cursor", 0))
