"""Loss terms reported for POWA style-transfer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class StyleTransferLoss:
    """Unweighted loss components and their weighted total."""

    content: torch.Tensor
    style: torch.Tensor
    regularization: torch.Tensor
    total: torch.Tensor


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Return the unnormalized per-image Gram matrix from Eq. S10."""

    if features.ndim != 4:
        raise ValueError("features must have shape [N, C, H, W]")
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    return torch.bmm(flattened, flattened.transpose(1, 2))


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    """Return the isotropic spatial sum specified in Eq. S11."""

    if image.ndim != 4 or image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("image must have shape [N, C, H, W] with H,W >= 2")
    vertical = image[..., 1:, :-1] - image[..., :-1, :-1]
    horizontal = image[..., :-1, 1:] - image[..., :-1, :-1]
    return torch.linalg.vector_norm(
        torch.stack((vertical, horizontal), dim=0), dim=0
    ).sum()


def style_transfer_loss(
    output_features: Sequence[torch.Tensor],
    content_features: Sequence[torch.Tensor],
    style_features: Sequence[torch.Tensor],
    output_image: torch.Tensor,
    *,
    content_layer: int = -1,
    style_layer_weights: Sequence[float] | None = None,
    content_weight: float = 1.0,
    style_weight: float = 1.0e6,
    regularization_weight: float = 1.0e-5,
) -> StyleTransferLoss:
    """Evaluate the squared feature norms and TV sum in Eqs. S8--S12.

    Feature extraction is caller-provided so the VGG-16 weights,
    preprocessing and returned feature layers can be selected explicitly.
    """

    if not output_features or len(output_features) != len(style_features):
        raise ValueError(
            "output_features and style_features must be non-empty and aligned"
        )
    return style_transfer_loss_from_grams(
        output_features,
        content_features,
        [gram_matrix(features) for features in style_features],
        output_image,
        content_layer=content_layer,
        style_layer_weights=style_layer_weights,
        content_weight=content_weight,
        style_weight=style_weight,
        regularization_weight=regularization_weight,
    )


def style_transfer_loss_from_grams(
    output_features: Sequence[torch.Tensor],
    content_features: Sequence[torch.Tensor],
    style_gram_targets: Sequence[torch.Tensor],
    output_image: torch.Tensor,
    *,
    content_layer: int = -1,
    style_layer_weights: Sequence[float] | None = None,
    content_weight: float = 1.0,
    style_weight: float = 1.0e6,
    regularization_weight: float = 1.0e-5,
) -> StyleTransferLoss:
    """Evaluate Eqs. S8--S12 using precomputed style Gram matrices."""

    if not output_features or len(output_features) != len(style_gram_targets):
        raise ValueError(
            "output features and style Gram targets must be non-empty and aligned"
        )
    if len(content_features) != len(output_features):
        raise ValueError("content_features must align with output_features")
    if style_layer_weights is None:
        style_layer_weights = [1.0] * len(output_features)
    if len(style_layer_weights) != len(output_features):
        raise ValueError("one style weight is required per feature layer")

    content = (
        output_features[content_layer] - content_features[content_layer]
    ).square().sum()
    style = output_image.new_zeros(())
    for weight, output, target_gram in zip(
        style_layer_weights, output_features, style_gram_targets, strict=True
    ):
        style = style + float(weight) * (
            gram_matrix(output) - target_gram
        ).square().sum()
    regularization = total_variation_loss(output_image)
    total = (
        float(content_weight) * content
        + float(style_weight) * style
        + float(regularization_weight) * regularization
    )
    return StyleTransferLoss(content, style, regularization, total)
