"""Command-line entry points for training, inference and parameter reporting."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from ..codebook import load_codebook
from .config import StyleTrainingConfig
from .data import load_style_references
from .inference import stylize_directory
from .model import POWAStyleTransferNetwork, style_parameter_statistics
from .perceptual import create_vgg16_feature_extractor
from .training import (
    StyleTransferTrainer,
    create_content_loader,
    load_style_model_state,
)


def _weights_argument(value: str) -> str | Path | None:
    return None if value.lower() == "none" else ("DEFAULT" if value == "DEFAULT" else Path(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="POWA optical style transfer")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train the 15 optical style banks")
    train.add_argument("--codebook", required=True, type=Path)
    train.add_argument("--content-dir", required=True, type=Path)
    train.add_argument("--style-dir", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--vgg-weights", default="DEFAULT", type=_weights_argument)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--workers", default=0, type=int)
    train.add_argument(
        "--train-encoder",
        action="store_true",
        help="also optimize the encoder (the default keeps it fixed)",
    )

    infer = commands.add_parser("infer", help="apply a trained style bank")
    infer.add_argument("--codebook", required=True, type=Path)
    infer.add_argument("--checkpoint", required=True, type=Path)
    infer.add_argument("--input-dir", required=True, type=Path)
    infer.add_argument("--output-dir", required=True, type=Path)
    selection = infer.add_mutually_exclusive_group(required=True)
    selection.add_argument("--style-id", type=int)
    selection.add_argument("--fusion-style-ids", nargs="+", type=int)
    infer.add_argument("--fusion-coefficients", nargs="+", type=float)
    infer.add_argument(
        "--expected-images", type=int, default=StyleTrainingConfig().test_images
    )
    infer.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    stats = commands.add_parser("stats", help="print exact model parameter counts")
    stats.add_argument("--codebook", required=True, type=Path)
    stats.add_argument(
        "--train-encoder",
        action="store_true",
        help="include the encoder in the trainable-parameter count",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    codebook = load_codebook(arguments.codebook)
    if arguments.command == "stats":
        model = POWAStyleTransferNetwork(
            codebook, freeze_encoder=not arguments.train_encoder
        )
        print(json.dumps(style_parameter_statistics(model).to_dict(), indent=2))
        return 0

    if arguments.command == "train":
        config = replace(
            StyleTrainingConfig(), freeze_encoder=not arguments.train_encoder
        )
        model = POWAStyleTransferNetwork(
            codebook, freeze_encoder=config.freeze_encoder
        )
        extractor = create_vgg16_feature_extractor(arguments.vgg_weights)
        references = load_style_references(arguments.style_dir, config=config)
        loader = create_content_loader(
            arguments.content_dir, config, workers=arguments.workers
        )
        trainer = StyleTransferTrainer(
            model, extractor, references, config=config, device=arguments.device
        )
        trainer.fit(loader, checkpoint_directory=arguments.output_dir)
        return 0

    model = POWAStyleTransferNetwork(codebook)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    load_style_model_state(model, checkpoint, codebook)
    stylize_directory(
        model,
        arguments.input_dir,
        arguments.output_dir,
        style_id=arguments.style_id,
        fusion_style_ids=(
            tuple(arguments.fusion_style_ids) if arguments.fusion_style_ids else None
        ),
        fusion_coefficients=(
            tuple(arguments.fusion_coefficients)
            if arguments.fusion_coefficients
            else None
        ),
        expected_images=arguments.expected_images,
        device=arguments.device,
    )
    return 0
