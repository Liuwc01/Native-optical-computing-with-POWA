"""Deterministic CPU demo of SOST interpolation and deployment rounding.

The example uses only the small simulated input file shipped beside this
script. Its five-point response bank is generated analytically and is not a
measured, task-specific, or manuscript-result codebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from powa import SOSTConv2d, WPUCodebook

DEFAULT_INPUT_PATH = Path(__file__).with_name("data") / "sost_demo_inputs.json"


def load_simulated_inputs(path: str | Path = DEFAULT_INPUT_PATH) -> torch.Tensor:
    """Load and validate the bundled simulated inputs."""

    with Path(path).open(encoding="utf-8") as handle:
        record = json.load(handle)
    shape = tuple(int(value) for value in record["shape"])
    if shape != (2, 1, 4, 4):
        raise ValueError(f"expected simulated input shape (2, 1, 4, 4), got {shape}")
    inputs = torch.tensor(record["values"], dtype=torch.float32)
    if inputs.numel() != 32 or not bool(torch.isfinite(inputs).all()):
        raise ValueError("the simulated input values must contain 32 finite numbers")
    return inputs.reshape(shape)


def synthetic_response_bank() -> WPUCodebook:
    """Create a five-point signed response bank solely for this demo."""

    centered_grid = torch.linspace(-1.0, 1.0, 5, dtype=torch.float32)
    port_scales = ((torch.arange(8, dtype=torch.float32) + 2.0) / 8.0).unsqueeze(1)
    weights = port_scales * centered_grid.unsqueeze(0)
    wavelength_nm = torch.linspace(1000.0, 1000.8, 5, dtype=torch.float64)
    return WPUCodebook(weights=weights, wavelength_nm=wavelength_nm)


def _six_decimal(value: torch.Tensor) -> float:
    return float(f"{value.detach().cpu().item():.6f}")


def run_demo(input_path: str | Path = DEFAULT_INPUT_PATH) -> dict[str, Any]:
    """Run one continuous and one rounded forward pass on CPU."""

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        inputs = load_simulated_inputs(input_path)
        layer = SOSTConv2d(
            synthetic_response_bank(),
            in_channels=1,
            out_channels=8,
            bias=False,
        ).cpu()
        with torch.no_grad():
            layer.wavelength_u.copy_(
                torch.tensor([[[0.0, 0.25, 0.625, 1.0]]], dtype=torch.float32)
            )

        layer.set_deployment(False)
        continuous_output = layer(inputs)
        loss = continuous_output.square().mean()
        loss.backward()
        gradient_norm = layer.wavelength_u.grad.norm()

        layer.set_deployment(True)
        with torch.no_grad():
            rounded_output = layer(inputs)

        summary = {
            "continuous_checksum": _six_decimal(continuous_output.sum()),
            "continuous_output_shape": list(continuous_output.shape),
            "device": "cpu",
            "gradient_norm": _six_decimal(gradient_norm),
            "input_shape": list(inputs.shape),
            "max_abs_rounding_difference": _six_decimal(
                (continuous_output - rounded_output).abs().max()
            ),
            "rounded_checksum": _six_decimal(rounded_output.sum()),
            "rounded_indices": layer.rounded_indices().flatten().tolist(),
            "rounded_output_shape": list(rounded_output.shape),
            "status": "ok",
        }
        if gradient_norm <= 0 or summary["max_abs_rounding_difference"] <= 0:
            raise RuntimeError("the demo did not exercise interpolation and rounding")
        return summary
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
