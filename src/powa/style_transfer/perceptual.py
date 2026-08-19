"""Injectable VGG-16 feature extraction for perceptual losses."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn


class VGG16FeatureExtractor(nn.Module):
    """Return VGG-16 convolution outputs numbered from one.

    The feature module is injected so callers may use torchvision-downloaded
    weights, a locally licensed state dictionary, or another compatible source.
    """

    def __init__(
        self,
        features: nn.Sequential,
        layers: tuple[int, ...] = (2, 4, 6, 9),
    ) -> None:
        super().__init__()
        if not layers or any(layer < 1 for layer in layers):
            raise ValueError("VGG convolution layer numbers must be positive")
        self.features = deepcopy(features)
        for module in self.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.layers = tuple(sorted(set(layers)))
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = (images - self.mean) / self.std
        selected: list[torch.Tensor] = []
        convolution_index = 0
        for layer in self.features:
            output = layer(output)
            if isinstance(layer, nn.Conv2d):
                convolution_index += 1
                if convolution_index in self.layers:
                    selected.append(output)
                if convolution_index >= self.layers[-1]:
                    break
        if len(selected) != len(self.layers):
            raise ValueError(
                f"feature module exposes only {convolution_index} convolutional layers"
            )
        return tuple(selected)


def create_vgg16_feature_extractor(
    weights: str | Path | None = "DEFAULT",
    *,
    layers: tuple[int, ...] = (2, 4, 6, 9),
) -> VGG16FeatureExtractor:
    """Create VGG-16 features from downloadable or caller-supplied weights.

    ``"DEFAULT"`` asks torchvision to download/cache its published weights.
    ``None`` creates an uninitialized model for structural testing only.  A
    filesystem path loads a caller-provided VGG-16 state dictionary.
    """

    from torchvision.models import VGG16_Weights, vgg16

    if weights == "DEFAULT":
        model = vgg16(weights=VGG16_Weights.DEFAULT)
    elif weights is None:
        model = vgg16(weights=None)
    else:
        model = vgg16(weights=None)
        state = torch.load(Path(weights), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return VGG16FeatureExtractor(model.features, layers=layers)
