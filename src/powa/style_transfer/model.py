"""Executable encoder--optical-style-bank--decoder network."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..codebook import WPUCodebook
from ..sost import SOSTConv2d, project_sost_wavelengths, set_sost_deployment

PAPER_STYLE_COUNT = 15


class RightBottomPad2d(nn.Module):
    """Add one zero-valued pixel to the right and bottom edges."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.pad(inputs, (0, 1, 0, 1))


class ImageEncoder(nn.Module):
    """Fully convolutional image encoder used before the latent style bank."""

    def __init__(
        self,
        latent_channels: int = 256,
        channels: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        specifications = (
            (3, c1, 9, 2, 4),
            (c1, c2, 3, 2, 1),
            (c2, c3, 3, 1, 1),
            (c3, latent_channels, 3, 1, 1),
        )
        layers: list[nn.Module] = []
        for input_channels, output_channels, kernel, stride, padding in specifications:
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(output_channels, affine=False),
                    nn.ReLU(inplace=False),
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ImageDecoder(nn.Module):
    """Fully convolutional decoder that restores the input image dimensions."""

    def __init__(
        self,
        latent_channels: int = 256,
        channels: tuple[int, int, int] = (128, 64, 32),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, c1, 3, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm2d(c1, affine=False),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(c1, c2, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(c2, affine=False),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(
                c2, c3, 3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.InstanceNorm2d(c3, affine=False),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(
                c3, 3, 9, stride=2, padding=4, output_padding=1, bias=False
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class OpticalStyleBank(nn.Module):
    """Four 2 x 2 wavelength-parameterized layers for one style."""

    def __init__(self, codebook: WPUCodebook, channels: int = 256) -> None:
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError("channels must be a positive multiple of eight")
        self.layers = nn.ModuleList(
            [SOSTConv2d(codebook, channels, channels, bias=False) for _ in range(4)]
        )
        self.normalizations = nn.ModuleList(
            [nn.InstanceNorm2d(channels, affine=False) for _ in range(4)]
        )
        self.padding = RightBottomPad2d()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, normalization in zip(self.layers, self.normalizations, strict=True):
            output = self.padding(output)
            output = F.relu(normalization(layer(output)), inplace=False)
        return output

    def get_extra_state(self) -> dict[str, bool]:
        """Persist discrete-deployment mode alongside wavelength parameters."""

        return {"deployment": all(layer.deployment for layer in self.layers)}

    def set_extra_state(self, state: object) -> None:
        enabled = bool(state.get("deployment", False)) if isinstance(state, dict) else False
        set_sost_deployment(self, enabled)


class OpticalStyleBankCollection(nn.Module):
    """A separately trainable optical bank for every configured style."""

    def __init__(
        self,
        codebook: WPUCodebook,
        styles: int = PAPER_STYLE_COUNT,
        channels: int = 256,
    ) -> None:
        super().__init__()
        if styles < 1:
            raise ValueError("styles must be positive")
        self.banks = nn.ModuleList(
            [OpticalStyleBank(codebook, channels=channels) for _ in range(styles)]
        )

    def __len__(self) -> int:
        return len(self.banks)

    def forward(
        self,
        inputs: torch.Tensor,
        style_ids: int | Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(style_ids, int):
            identifiers = [style_ids] * inputs.shape[0]
        elif isinstance(style_ids, torch.Tensor):
            identifiers = [int(value) for value in style_ids.detach().cpu().flatten()]
        else:
            identifiers = [int(value) for value in style_ids]
        if len(identifiers) != inputs.shape[0]:
            raise ValueError("one style id is required per input image")
        if any(identifier < 0 or identifier >= len(self) for identifier in identifiers):
            raise IndexError("style id is outside the configured style-bank range")
        return torch.cat(
            [
                self.banks[identifier](sample.unsqueeze(0))
                for sample, identifier in zip(inputs, identifiers, strict=True)
            ],
            dim=0,
        )

    def fused(
        self,
        inputs: torch.Tensor,
        style_ids: Sequence[int],
        coefficients: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        selected = [self.banks[int(identifier)] for identifier in style_ids]
        return fuse_style_banks(inputs, selected, coefficients)


class POWAStyleTransferNetwork(nn.Module):
    """Complete image-to-image network with latent optical style banks."""

    def __init__(
        self,
        codebook: WPUCodebook,
        *,
        styles: int = PAPER_STYLE_COUNT,
        latent_channels: int = 256,
        encoder_channels: tuple[int, int, int] = (32, 64, 128),
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ImageEncoder(latent_channels, encoder_channels)
        self.style_banks = OpticalStyleBankCollection(
            codebook, styles=styles, channels=latent_channels
        )
        self.decoder = ImageDecoder(latent_channels, tuple(reversed(encoder_channels)))
        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, enabled: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(enabled)

    @staticmethod
    def _pad_to_multiple_of_four(inputs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = inputs.shape[-2:]
        bottom = (-height) % 4
        right = (-width) % 4
        if bottom or right:
            inputs = F.pad(inputs, (0, right, 0, bottom), mode="replicate")
        return inputs, height, width

    def forward(
        self,
        inputs: torch.Tensor,
        style_ids: int | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        padded, height, width = self._pad_to_multiple_of_four(inputs)
        latent = self.encoder(padded)
        if style_ids is not None:
            latent = self.style_banks(latent, style_ids)
        output = self.decoder(latent)
        return output[..., :height, :width]

    def forward_fused(
        self,
        inputs: torch.Tensor,
        style_ids: Sequence[int],
        coefficients: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        padded, height, width = self._pad_to_multiple_of_four(inputs)
        latent = self.encoder(padded)
        latent = self.style_banks.fused(latent, style_ids, coefficients)
        output = self.decoder(latent)
        return output[..., :height, :width]


def fuse_style_banks(
    inputs: torch.Tensor,
    banks: Sequence[OpticalStyleBank],
    coefficients: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Apply a convex, layer-wise mixture of independently trained kernels."""

    if not banks:
        raise ValueError("at least one style bank is required")
    if len(banks) != len(coefficients):
        raise ValueError("one coefficient is required per style bank")
    weights = torch.as_tensor(coefficients, device=inputs.device, dtype=inputs.dtype)
    if bool((weights < 0).any()) or not torch.isclose(
        weights.sum(), weights.new_tensor(1.0), atol=1e-6
    ):
        raise ValueError("coefficients must be nonnegative and sum to one")
    if any(len(bank.layers) != 4 for bank in banks):
        raise ValueError("each optical style bank must have four layers")

    output = inputs
    reference = banks[0]
    for layer_index, normalization in enumerate(reference.normalizations):
        fused_weight = sum(
            coefficient * bank.layers[layer_index].current_weight()
            for bank, coefficient in zip(banks, weights, strict=True)
        )
        output = reference.padding(output)
        output = F.conv2d(output, fused_weight)
        output = F.relu(normalization(output), inplace=False)
    return output


@torch.no_grad()
def finalize_style_network(model: nn.Module) -> None:
    """Project learned variables and enable persistent grid deployment."""

    project_sost_wavelengths(model)
    set_sost_deployment(model, True)


@torch.no_grad()
def deployment_wavelengths_nm(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return rounded laser wavelengths for all named optical layers."""

    return {
        name: layer.selected_wavelengths_nm()
        for name, layer in model.named_modules()
        if isinstance(layer, SOSTConv2d)
    }


@dataclass(frozen=True)
class StyleParameterStatistics:
    """Exact parameter counts for a concrete style-transfer network.

    In the default 15-bank configuration, the optical banks contain
    15 * 4 * 2 * 2 * 256 * 256 / 8 = 1,966,080 wavelength parameters, and the
    decoder contains 394,848 parameters. With the encoder fixed, the
    optical-bank share of the optimized network is
    1,966,080 / (1,966,080 + 394,848) = 83.2757%, or approximately 84% of the
    optimized parameters. Fixed encoder parameters are excluded from this
    denominator.
    """

    total_parameters: int
    trainable_parameters: int
    encoder_parameters: int
    decoder_parameters: int
    style_bank_parameters: int
    trainable_style_bank_parameters: int
    fixed_encoder_trainable_parameters: int
    style_bank_share_of_total: float
    style_bank_share_of_trainable: float
    style_bank_share_with_fixed_encoder: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def style_parameter_statistics(model: POWAStyleTransferNetwork) -> StyleParameterStatistics:
    """Count wavelength and network parameters without approximation.

    ``style_bank_share_with_fixed_encoder`` is all style-bank wavelength
    parameters divided by the sum of style-bank and decoder parameters. It is
    the approximately 84% optimized-parameter share when the encoder is fixed.
    """

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    decoder = sum(parameter.numel() for parameter in model.decoder.parameters())
    bank = sum(parameter.numel() for parameter in model.style_banks.parameters())
    trainable_bank = sum(
        parameter.numel()
        for parameter in model.style_banks.parameters()
        if parameter.requires_grad
    )
    return StyleParameterStatistics(
        total_parameters=total,
        trainable_parameters=trainable,
        encoder_parameters=encoder,
        decoder_parameters=decoder,
        style_bank_parameters=bank,
        trainable_style_bank_parameters=trainable_bank,
        fixed_encoder_trainable_parameters=bank + decoder,
        style_bank_share_of_total=bank / total if total else 0.0,
        style_bank_share_of_trainable=trainable_bank / trainable if trainable else 0.0,
        style_bank_share_with_fixed_encoder=(
            bank / (bank + decoder) if bank + decoder else 0.0
        ),
    )
