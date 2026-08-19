"""Differentiable wavelength-indexed convolution used by SOST."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import WPUCodebook


class SOSTConv2d(nn.Module):
    """Build eight-output 2 × 2 kernels from four wavelength parameters.

    Continuous mode linearly interpolates adjacent measured response columns.
    Hard-STE mode selects the nearest column in the forward pass and uses the
    interpolation gradient in the backward pass. Deployment selects the exact
    nearest grid column. `wavelength_u` must be projected into `[0, 1]` after
    each optimizer update.
    """

    def __init__(
        self,
        codebook: WPUCodebook,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = False,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        training_mode: str = "continuous",
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 8 or out_channels % 8:
            raise ValueError(
                "in_channels must be positive and out_channels a positive multiple of 8"
            )
        self.register_buffer("codebook", codebook.weights.detach().clone())
        self.register_buffer("wavelength_nm", codebook.wavelength_nm.detach().clone())
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups_of_eight = self.out_channels // 8
        self.grid_points = codebook.points
        if training_mode not in {"continuous", "hard_ste"}:
            raise ValueError("training_mode must be 'continuous' or 'hard_ste'")
        self.training_mode = training_mode
        self.deployment = False
        self.wavelength_u = nn.Parameter(
            torch.empty(self.groups_of_eight, self.in_channels, 4)
        )
        nn.init.uniform_(self.wavelength_u, 0.0, 1.0)
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (
            (padding, padding) if isinstance(padding, int) else tuple(padding)
        )
        self.dilation = (
            (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        )

    def continuous_indices(self) -> torch.Tensor:
        return (self.grid_points - 1) * self.wavelength_u.clamp(0.0, 1.0)

    def rounded_indices(self) -> torch.Tensor:
        # Deterministic nearest neighbor; exact half-grid ties go upward.
        return (
            torch.floor(self.continuous_indices() + 0.5)
            .long()
            .clamp(0, self.grid_points - 1)
        )

    def _sample_continuous(self) -> torch.Tensor:
        indices = self.continuous_indices()
        lower = torch.floor(indices).long().clamp(0, self.grid_points - 1)
        upper = (lower + 1).clamp(max=self.grid_points - 1)
        fraction = (indices - lower.to(indices.dtype)).unsqueeze(-1)
        bank = self.codebook.t()
        return (1.0 - fraction) * bank[lower] + fraction * bank[upper]

    def _sample_rounded(self) -> torch.Tensor:
        return self.codebook.t()[self.rounded_indices()]

    def current_weight(self) -> torch.Tensor:
        continuous = self._sample_continuous()
        if self.deployment:
            selected = self._sample_rounded()
        elif self.training_mode == "hard_ste":
            rounded = self._sample_rounded()
            selected = continuous + (rounded - continuous).detach()
        else:
            selected = continuous
        return (
            selected.permute(0, 3, 1, 2)
            .contiguous()
            .view(self.out_channels, self.in_channels, 2, 2)
        )

    def set_deployment(self, enabled: bool = True) -> None:
        self.deployment = bool(enabled)

    @torch.no_grad()
    def project_wavelengths(self) -> None:
        self.wavelength_u.clamp_(0.0, 1.0)

    @torch.no_grad()
    def selected_wavelengths_nm(self) -> torch.Tensor:
        return self.wavelength_nm[self.rounded_indices()].cpu()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected inputs [N, {self.in_channels}, H, W], got {tuple(inputs.shape)}"
            )
        return F.conv2d(
            inputs,
            self.current_weight(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


def set_sost_deployment(module: nn.Module, enabled: bool = True) -> None:
    for child in module.modules():
        if isinstance(child, SOSTConv2d):
            child.set_deployment(enabled)


@torch.no_grad()
def project_sost_wavelengths(module: nn.Module) -> None:
    """Project every SOST wavelength parameter back into `u ∈ [0, 1]`."""

    for child in module.modules():
        if isinstance(child, SOSTConv2d):
            child.project_wavelengths()


@torch.no_grad()
def validate_sost_codebook_buffers(
    module: nn.Module,
    reference: WPUCodebook,
) -> int:
    """Require every loaded SOST buffer to equal the selected physical bank."""

    expected_weights = reference.weights.detach().cpu()
    expected_wavelengths = reference.wavelength_nm.detach().cpu()
    checked = 0
    for name, child in module.named_modules():
        if not isinstance(child, SOSTConv2d):
            continue
        checked += 1
        observed_weights = child.codebook.detach().cpu()
        observed_wavelengths = child.wavelength_nm.detach().cpu()
        if not torch.equal(observed_weights, expected_weights):
            raise ValueError(
                f"SOST codebook buffer in layer {name!r} does not match "
                "the selected physical codebook"
            )
        if not torch.equal(observed_wavelengths, expected_wavelengths):
            raise ValueError(
                f"SOST wavelength buffer in layer {name!r} does not match "
                "the selected physical grid"
            )
    if checked == 0:
        raise ValueError("model contains no SOSTConv2d layers to validate")
    return checked


@torch.no_grad()
def validate_sost_state_dict_codebook(
    state_dict: Mapping[str, torch.Tensor],
    reference: WPUCodebook,
) -> int:
    """Validate SOST codebook buffers in a state dict before model loading.

    PyTorch state dictionaries persist the codebook and wavelength grid for
    every SOST layer. Checking those tensors before ``load_state_dict`` avoids
    silently replacing the codebook selected by the caller.
    """

    codebook_buffers = {
        key.removesuffix("codebook"): value
        for key, value in state_dict.items()
        if key == "codebook" or key.endswith(".codebook")
    }
    wavelength_buffers = {
        key.removesuffix("wavelength_nm"): value
        for key, value in state_dict.items()
        if key == "wavelength_nm" or key.endswith(".wavelength_nm")
    }
    if not codebook_buffers:
        raise ValueError("checkpoint contains no SOST codebook buffers to validate")
    if codebook_buffers.keys() != wavelength_buffers.keys():
        raise ValueError("checkpoint has incomplete SOST codebook metadata")

    expected_weights = reference.weights.detach().cpu()
    expected_wavelengths = reference.wavelength_nm.detach().cpu()
    for prefix, observed_weights in codebook_buffers.items():
        observed_wavelengths = wavelength_buffers[prefix]
        if not isinstance(observed_weights, torch.Tensor) or not torch.equal(
            observed_weights.detach().cpu(), expected_weights
        ):
            raise ValueError(
                f"SOST codebook buffer in checkpoint layer {prefix!r} does not "
                "match the selected physical codebook"
            )
        if not isinstance(observed_wavelengths, torch.Tensor) or not torch.equal(
            observed_wavelengths.detach().cpu(), expected_wavelengths
        ):
            raise ValueError(
                f"SOST wavelength buffer in checkpoint layer {prefix!r} does not "
                "match the selected physical grid"
            )
    return len(codebook_buffers)
