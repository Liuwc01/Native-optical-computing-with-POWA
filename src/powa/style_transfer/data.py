"""Image discovery and paper-aligned style-transfer transforms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import StyleTrainingConfig

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def discover_images(root: str | Path) -> list[Path]:
    """Return image files below ``root`` in stable lexical order."""

    directory = Path(root).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no supported images found below {directory}")
    return paths


class ConvertRGB:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


class ResizeLongSide:
    """Resize while preserving aspect ratio so the longer side equals ``size``."""

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        return image.resize(resized, resample=Image.Resampling.BILINEAR)


def content_training_transform(
    config: StyleTrainingConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """Convert to RGB and take the reported random 512 x 512 content crop."""

    settings = config or StyleTrainingConfig()
    return transforms.Compose(
        (
            ConvertRGB(),
            transforms.RandomCrop(settings.content_crop_pixels),
            transforms.ToTensor(),
        )
    )


def style_reference_transform(
    config: StyleTrainingConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """Convert to RGB and scale the longer side to the reported 600 pixels."""

    settings = config or StyleTrainingConfig()
    return transforms.Compose(
        (
            ConvertRGB(),
            ResizeLongSide(settings.style_long_side_pixels),
            transforms.ToTensor(),
        )
    )


def inference_transform() -> Callable[[Image.Image], torch.Tensor]:
    """Preserve the full RGB image dimensions for model-side pad and crop."""

    return transforms.Compose((ConvertRGB(), transforms.ToTensor()))


class ImagePathDataset(Dataset[torch.Tensor]):
    """Dataset for a flat or nested directory of ordinary image files."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        expected_count: int | None = None,
    ) -> None:
        self.paths = discover_images(root)
        if expected_count is not None and len(self.paths) != expected_count:
            raise ValueError(
                f"expected {expected_count} images below {root}, found {len(self.paths)}"
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def load_style_references(
    root: str | Path,
    *,
    config: StyleTrainingConfig | None = None,
    expected_count: int | None = None,
) -> list[torch.Tensor]:
    """Load style references individually; aspect ratios may produce distinct shapes."""

    settings = config or StyleTrainingConfig()
    dataset = ImagePathDataset(
        root,
        style_reference_transform(settings),
        expected_count=settings.styles if expected_count is None else expected_count,
    )
    return [dataset[index] for index in range(len(dataset))]


def validate_style_ids(style_ids: Sequence[int], styles: int) -> None:
    if any(identifier < 0 or identifier >= styles for identifier in style_ids):
        raise IndexError(f"style identifiers must lie in [0, {styles - 1}]")
