"""Full-resolution single-style and fused-style inference."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .data import discover_images, inference_transform
from .model import POWAStyleTransferNetwork, finalize_style_network


@torch.no_grad()
def stylize_tensor(
    model: POWAStyleTransferNetwork,
    image: torch.Tensor,
    *,
    style_id: int | None = None,
    fusion_style_ids: tuple[int, ...] | None = None,
    fusion_coefficients: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Stylize one ``[3,H,W]`` or ``[1,3,H,W]`` tensor."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must have shape [3,H,W] or [1,3,H,W]")
    if style_id is not None and fusion_style_ids is not None:
        raise ValueError("select either one style or a fusion, not both")
    if fusion_style_ids is not None:
        if fusion_coefficients is None:
            raise ValueError("fusion coefficients are required")
        output = model.forward_fused(image, fusion_style_ids, fusion_coefficients)
    elif style_id is not None:
        output = model(image, style_id)
    else:
        raise ValueError("a style id or fusion specification is required")
    return output.squeeze(0).clamp(0.0, 1.0)


def stylize_directory(
    model: POWAStyleTransferNetwork,
    input_directory: str | Path,
    output_directory: str | Path,
    *,
    style_id: int | None = None,
    fusion_style_ids: tuple[int, ...] | None = None,
    fusion_coefficients: tuple[float, ...] | None = None,
    expected_images: int | None = None,
    device: torch.device | str = "cpu",
) -> list[Path]:
    """Run deterministic deployment inference over a directory of images."""

    paths = discover_images(input_directory)
    if expected_images is not None and len(paths) != expected_images:
        raise ValueError(f"expected {expected_images} inputs, found {len(paths)}")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    finalize_style_network(model)
    transform = inference_transform()
    written: list[Path] = []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            tensor = transform(source).to(device)
        result = stylize_tensor(
            model,
            tensor,
            style_id=style_id,
            fusion_style_ids=fusion_style_ids,
            fusion_coefficients=fusion_coefficients,
        )
        target = destination / f"{index:03d}_{path.stem}.png"
        to_pil_image(result.cpu()).save(target)
        written.append(target)
    return written
