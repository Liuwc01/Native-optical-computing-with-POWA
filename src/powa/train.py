"""Train the O-VGG classifier with seeded public defaults."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import torch
import torchvision

from .codebook import load_codebook
from .data import get_train_and_test_loaders
from .model import OVGGClassifier
from .sost import set_sost_deployment
from .training import TrainingConfig, make_optimizer, run_epoch, set_seed


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument(
        "--codebook",
        type=Path,
        required=True,
        help="path to a prepared 8 x K WPU codebook archive on a 0.2-nm grid",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    config = TrainingConfig(seed=args.seed, num_workers=args.num_workers)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codebook = load_codebook(args.codebook)
    train_loader, test_loader, num_classes = get_train_and_test_loaders(
        args.dataset,
        args.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    model = OVGGClassifier(codebook, num_classes=num_classes).to(device)
    set_sost_deployment(model, False)
    optimizer = make_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=0.0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, config.epochs + 1):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            label_smoothing=config.label_smoothing,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        # Record an evaluation curve at every epoch. The command completes the
        # full schedule and writes checkpoint_last at epoch 400.
        test_loss, test_accuracy = run_epoch(model, test_loader, device)
        history.append(
            (
                epoch,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                learning_rate_used,
            )
        )
        scheduler.step()
        print(
            "epoch=%03d/%d train_loss=%.6f train_accuracy=%.4f "
            "test_loss=%.6f test_accuracy=%.4f"
            % (
                epoch,
                config.epochs,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
            )
        )

    checkpoint = {
        "model": model.state_dict(),
        "dataset": args.dataset,
        "num_classes": num_classes,
        "architecture": "OVGGBackbone-13x2x2conv-BN-5maxpool",
        "head_variant": model.head_variant,
        "sost_training_mode": model.sost_training_mode,
        "deployment_rounding": "nearest; exact half-grid ties round upward",
        "epochwise_evaluation": (
            "official_test_split_reporting_curve; checkpoint_last_at_epoch_400"
        ),
        "training_config": config.to_dict(),
        "codebook_fingerprint": codebook.fingerprint,
        "codebook_grid": {
            "points": codebook.points,
            "start_nm": float(codebook.wavelength_nm[0]),
            "stop_nm": float(codebook.wavelength_nm[-1]),
            "interval_nm": float(codebook.wavelength_nm[1] - codebook.wavelength_nm[0]),
        },
        "software_versions": {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda_runtime": None
            if torch.version.cuda is None
            else str(torch.version.cuda),
        },
        "epoch": config.epochs,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint_last.pth")
    with (args.output_dir / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "epoch",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
                "learning_rate",
            )
        )
        writer.writerows(history)

    # Re-evaluate the fixed 400th-epoch model after deployment-grid rounding.
    set_sost_deployment(model, True)
    test_loss, test_accuracy = run_epoch(model, test_loader, device)
    metrics = {
        "dataset": args.dataset,
        "mode": "rounded_deployment",
        "head_variant": model.head_variant,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_examples": len(test_loader.dataset),
        "checkpoint": "checkpoint_last.pth",
        "codebook_fingerprint": codebook.fingerprint,
    }
    with (args.output_dir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
