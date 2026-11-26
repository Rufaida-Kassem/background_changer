"""Train the person-segmentation model used by the background changer."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation_model import SegmentationModelFactory
from training_data import DatasetPairFinder, MaskAnalyzer, PersonDataset
from trainer import PersonSegmentationTrainer, TrainingConfig


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "deeplabv3_background_segmentation.pt"
IMAGE_SIZE = (256, 256)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepLabV3 to segment people from background."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing the image and mask folders.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Optional image directory when automatic pair discovery is unsuitable.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="Optional mask directory; use together with --images-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output TorchScript model path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize DeepLabV3 without downloading pretrained weights.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    options = parse_args(args)
    if (options.images_dir is None) != (options.masks_dir is None):
        raise ValueError("--images-dir and --masks-dir must be provided together.")
    if options.epochs < 1 or options.batch_size < 1 or options.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")
    if options.num_workers < 0 or options.sample_size < 1:
        raise ValueError("num-workers must be non-negative and sample-size positive.")
    if options.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")

    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.seed)

    pairs = DatasetPairFinder.find_pairs(
        options.dataset_root,
        image_directory=options.images_dir,
        mask_directory=options.masks_dir,
    )
    statistics = MaskAnalyzer.analyze(
        pairs,
        sample_size=options.sample_size,
        seed=options.seed,
    )
    print(f"Found {len(pairs)} image/mask pairs.")
    print(
        f"Mask values: {statistics.values[:10]}; "
        f"foreground threshold: {statistics.threshold:g}; "
        f"average person area: {statistics.average_person_fraction:.1%}."
    )
    print(
        f"Image/mask size mismatches in sample: "
        f"{statistics.size_mismatches}/{statistics.sample_count}."
    )

    pairs = list(pairs)
    random.Random(options.seed).shuffle(pairs)
    validation_count = max(1, int(0.15 * len(pairs)))
    if validation_count >= len(pairs):
        raise ValueError("At least two image/mask pairs are required for training.")
    validation_pairs = pairs[:validation_count]
    training_pairs = pairs[validation_count:]

    train_dataset = PersonDataset(
        training_pairs, IMAGE_SIZE, statistics.threshold
    )
    validation_dataset = PersonDataset(
        validation_pairs, IMAGE_SIZE, statistics.threshold
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            num_workers=options.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            pin_memory=device.type == "cuda",
        ),
    }

    print(
        f"Training on {len(training_pairs)} images; validating on "
        f"{len(validation_pairs)}. Device: {device}."
    )
    model = SegmentationModelFactory.create(pretrained=not options.no_pretrained)
    config = TrainingConfig(
        epochs=options.epochs,
        patience=options.patience,
        learning_rate=options.learning_rate,
    )
    trainer = PersonSegmentationTrainer(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        config=config,
    )
    trainer.fit(options.output)


if __name__ == "__main__":
    main()
