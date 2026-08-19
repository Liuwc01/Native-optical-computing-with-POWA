"""Standalone checkpoint evaluation in rounded deployment mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .codebook import load_codebook
from .data import get_test_loader
from .model import OVGGClassifier
from .sost import (
    set_sost_deployment,
    validate_sost_codebook_buffers,
    validate_sost_state_dict_codebook,
)
from .training import smoothed_cross_entropy


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--codebook",
        type=Path,
        required=True,
        help="path to the 8 x K uniform-grid WPU codebook used by the checkpoint",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    required = {
        "model",
        "dataset",
        "num_classes",
        "head_variant",
        "codebook_fingerprint",
        "codebook_grid",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing metadata: {sorted(missing)}")
    grid = checkpoint["codebook_grid"]
    if not isinstance(grid, dict) or "interval_nm" not in grid:
        raise ValueError("checkpoint has invalid codebook-grid metadata")
    codebook = load_codebook(
        args.codebook,
        expected_interval_nm=float(grid["interval_nm"]),
    )
    if checkpoint["codebook_fingerprint"] != codebook.fingerprint:
        raise ValueError("the codebook does not match the checkpoint fingerprint")
    if checkpoint["head_variant"] != OVGGClassifier.head_variant:
        raise ValueError("the checkpoint uses an unsupported classifier-head variant")
    dataset = str(checkpoint["dataset"])
    test_loader, expected_classes = get_test_loader(
        dataset,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if int(checkpoint["num_classes"]) != expected_classes:
        raise ValueError("checkpoint class count does not match its named dataset")
    model = OVGGClassifier(codebook, num_classes=expected_classes)
    validate_sost_state_dict_codebook(checkpoint["model"], codebook)
    model.load_state_dict(checkpoint["model"], strict=True)
    # The state dict itself contains per-layer codebook/grid buffers. Strict
    # loading can overwrite the buffers initialized from --codebook, so verify
    # every loaded layer before any inference.
    validate_sost_codebook_buffers(model, codebook)
    model.to(device)
    model.eval()
    set_sost_deployment(model, True)
    predictions = []
    targets = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss = smoothed_cross_entropy(logits, labels, smoothing=0.0)
            total_loss += float(loss) * int(labels.shape[0])
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels.cpu())
    predicted = torch.cat(predictions).numpy().astype(np.int64)
    target = torch.cat(targets).numpy().astype(np.int64)
    correct = int((predicted == target).sum())
    metrics = {
        "dataset": dataset,
        "mode": "rounded_deployment",
        "head_variant": model.head_variant,
        "correct": correct,
        "total": int(target.size),
        "accuracy": correct / float(target.size),
        "cross_entropy": total_loss / float(target.size),
        "codebook_fingerprint": codebook.fingerprint,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        predicted_class=predicted,
        target_class=target,
    )
    with (args.output_dir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
