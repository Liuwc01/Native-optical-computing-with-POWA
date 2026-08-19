"""O-VGG architecture matching the Figure 3 parameter configuration."""

from __future__ import annotations

import torch
import torch.nn as nn

from .codebook import WPUCodebook
from .sost import SOSTConv2d


class RightBottomPad2d(nn.Module):
    """Reflect one column to the right and one row to the bottom."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2] < 2 or inputs.shape[-1] < 2:
            raise ValueError("reflection padding requires height and width of at least 2")
        horizontal = torch.cat((inputs, inputs[..., -2:-1]), dim=-1)
        return torch.cat((horizontal, horizontal[..., -2:-1, :]), dim=-2)


def _optical_block(
    codebook: WPUCodebook,
    in_channels: int,
    out_channels: int,
    convolutions: int,
    *,
    batch_norm_epsilon: float,
    batch_norm_momentum: float,
    sost_training_mode: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = in_channels
    for _ in range(convolutions):
        layers.extend(
            [
                RightBottomPad2d(),
                SOSTConv2d(
                    codebook,
                    in_channels=current_channels,
                    out_channels=out_channels,
                    bias=False,
                    training_mode=sost_training_mode,
                ),
                nn.BatchNorm2d(
                    out_channels,
                    eps=batch_norm_epsilon,
                    momentum=batch_norm_momentum,
                ),
                nn.ReLU(inplace=True),
            ]
        )
        current_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class OVGGBackbone(nn.Module):
    """The 13-convolution, five-pool optical VGG backbone.

    BatchNorm epsilon and momentum are explicit and can be overridden when the
    model is constructed. The default training mode is hard-STE: nearest-grid
    weights in the forward pass and interpolation gradients in the backward
    pass.
    """

    def __init__(
        self,
        codebook: WPUCodebook,
        *,
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 0.9,
        sost_training_mode: str = "hard_ste",
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _optical_block(
                codebook,
                3,
                64,
                2,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                sost_training_mode=sost_training_mode,
            ),
            _optical_block(
                codebook,
                64,
                128,
                2,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                sost_training_mode=sost_training_mode,
            ),
            _optical_block(
                codebook,
                128,
                256,
                3,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                sost_training_mode=sost_training_mode,
            ),
            _optical_block(
                codebook,
                256,
                512,
                3,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                sost_training_mode=sost_training_mode,
            ),
            _optical_block(
                codebook,
                512,
                512,
                3,
                batch_norm_epsilon=batch_norm_epsilon,
                batch_norm_momentum=batch_norm_momentum,
                sost_training_mode=sost_training_mode,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs)


class OVGGClassifier(nn.Module):
    """O-VGG classifier with a two-FC classification head and hard-STE default."""

    head_variant = "gap_2fc"

    def __init__(
        self,
        codebook: WPUCodebook,
        num_classes: int,
        *,
        batch_norm_epsilon: float = 1e-5,
        batch_norm_momentum: float = 0.9,
        sost_training_mode: str = "hard_ste",
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.sost_training_mode = sost_training_mode
        self.backbone = OVGGBackbone(
            codebook,
            batch_norm_epsilon=batch_norm_epsilon,
            batch_norm_momentum=batch_norm_momentum,
            sost_training_mode=sost_training_mode,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        linear_layers = [
            layer for layer in self.classifier if isinstance(layer, nn.Linear)
        ]
        for index, layer in enumerate(linear_layers):
            nonlinearity = "relu" if index < len(linear_layers) - 1 else "linear"
            nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        pooled = self.avgpool(features)
        return self.classifier(torch.flatten(pooled, 1))
