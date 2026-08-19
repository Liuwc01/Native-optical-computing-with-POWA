"""Loading and physical-grid validation for the WPU optical weight bank."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class WPUCodebook:
    """Mean-centered optical weights paired with a physical wavelength axis."""

    weights: torch.Tensor
    wavelength_nm: torch.Tensor

    def __post_init__(self) -> None:
        weights = torch.as_tensor(self.weights, dtype=torch.float32).contiguous()
        wavelength = (
            torch.as_tensor(self.wavelength_nm, dtype=torch.float64)
            .flatten()
            .contiguous()
        )
        if weights.ndim != 2 or weights.shape[0] != 8:
            raise ValueError("weights must have shape [8, K]")
        if wavelength.shape != (weights.shape[1],):
            raise ValueError("wavelength_nm must have shape [K]")
        if not bool(torch.isfinite(weights).all()) or not bool(
            torch.isfinite(wavelength).all()
        ):
            raise ValueError("the codebook contains non-finite values")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "wavelength_nm", wavelength)

    @property
    def points(self) -> int:
        return int(self.weights.shape[1])

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.weights.cpu().numpy().astype("<f4").tobytes())
        digest.update(self.wavelength_nm.cpu().numpy().astype("<f8").tobytes())
        return digest.hexdigest()


def load_codebook(
    path: str | Path,
    *,
    expected_interval_nm: float = 0.2,
) -> WPUCodebook:
    """Load an explicitly supplied, uniformly sampled codebook archive."""

    with np.load(Path(path), allow_pickle=False) as archive:
        normalized = np.asarray(archive["codebook"], dtype=np.float32)
        wavelength_nm = np.asarray(archive["wavelength_nm"], dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[0] != 8 or normalized.shape[1] < 2:
        raise ValueError(
            f"expected codebook shape (8, K) with K >= 2, got {normalized.shape}"
        )
    points = normalized.shape[1]
    if wavelength_nm.shape != (points,):
        raise ValueError(f"expected wavelength_nm shape ({points},)")
    intervals = np.diff(wavelength_nm)
    if not np.isfinite(intervals).all() or not np.all(intervals > 0.0):
        raise ValueError("the codebook wavelength grid must be finite and increasing")
    if expected_interval_nm <= 0.0 or not np.isfinite(expected_interval_nm):
        raise ValueError("expected_interval_nm must be finite and positive")
    if not np.allclose(intervals, expected_interval_nm, rtol=0.0, atol=1e-7):
        raise ValueError(
            "the codebook wavelength grid must use uniform "
            f"{expected_interval_nm:g}-nm intervals"
        )
    if not np.isfinite(normalized).all():
        raise ValueError("the codebook contains non-finite values")
    if not np.allclose(normalized.min(axis=1), 0.0, atol=1e-6) or not np.allclose(
        normalized.max(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("each normalized output-port response must span [0, 1]")
    weights = torch.from_numpy(normalized)
    weights = weights - weights.mean(dim=1, keepdim=True)
    standard_deviation = weights.std(dim=1, keepdim=True, correction=1)
    if bool((standard_deviation <= 0.0).any()):
        raise ValueError("each codebook row must have nonzero variation")
    weights = weights / (standard_deviation + 1e-6) * 0.20
    return WPUCodebook(weights=weights, wavelength_nm=wavelength_nm)
