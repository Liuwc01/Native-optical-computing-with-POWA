"""Wavelength-parameterized optical style-bank building blocks."""

from .config import StyleTrainingConfig
from .data import (
    ImagePathDataset,
    ResizeLongSide,
    content_training_transform,
    inference_transform,
    load_style_references,
    style_reference_transform,
)
from .inference import stylize_directory, stylize_tensor
from .losses import (
    StyleTransferLoss,
    gram_matrix,
    style_transfer_loss,
    style_transfer_loss_from_grams,
    total_variation_loss,
)
from .model import (
    PAPER_STYLE_COUNT,
    ImageDecoder,
    ImageEncoder,
    OpticalStyleBank,
    OpticalStyleBankCollection,
    POWAStyleTransferNetwork,
    StyleParameterStatistics,
    deployment_wavelengths_nm,
    finalize_style_network,
    fuse_style_banks,
    style_parameter_statistics,
)
from .perceptual import VGG16FeatureExtractor, create_vgg16_feature_extractor
from .training import (
    EpochMetrics,
    StyleTransferTrainer,
    create_content_loader,
    create_style_optimizer,
    create_style_scheduler,
    load_style_model_state,
)

__all__ = [
    "PAPER_STYLE_COUNT",
    "EpochMetrics",
    "ImageDecoder",
    "ImageEncoder",
    "ImagePathDataset",
    "OpticalStyleBank",
    "OpticalStyleBankCollection",
    "POWAStyleTransferNetwork",
    "ResizeLongSide",
    "StyleParameterStatistics",
    "StyleTransferLoss",
    "StyleTransferTrainer",
    "StyleTrainingConfig",
    "VGG16FeatureExtractor",
    "content_training_transform",
    "create_content_loader",
    "create_style_optimizer",
    "create_style_scheduler",
    "create_vgg16_feature_extractor",
    "deployment_wavelengths_nm",
    "finalize_style_network",
    "fuse_style_banks",
    "gram_matrix",
    "inference_transform",
    "load_style_references",
    "load_style_model_state",
    "style_parameter_statistics",
    "style_reference_transform",
    "style_transfer_loss",
    "style_transfer_loss_from_grams",
    "stylize_directory",
    "stylize_tensor",
    "total_variation_loss",
]
