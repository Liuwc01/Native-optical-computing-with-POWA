"""Style-transfer configuration reported in the Supplementary Methods."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleTrainingConfig:
    """Machine-readable settings for the manuscript-associated experiment.

    ``weight_decay`` is explicit and defaults to PyTorch's AdamW default.  The
    complete setting is recorded in training checkpoints.
    """

    styles: int = 15
    content_crop_pixels: int = 512
    style_long_side_pixels: int = 600
    test_images: int = 25
    epochs: int = 150
    optimizer: str = "AdamW"
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    decay_factor: float = 0.8
    decay_every_epochs: int = 10
    freeze_encoder: bool = True
    content_weight: float = 1.0
    style_weight: float = 1.0e6
    regularization_weight: float = 1.0e-5
    latent_channels: int = 256
    content_vgg_layer: int = 9
    style_vgg_layers: tuple[int, ...] = (2, 4, 6, 9)
    style_layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
