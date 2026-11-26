"""Dataset discovery, validation, and preprocessing for person segmentation."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ImageMaskPair = tuple[Path, Path]
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MASK_NAME_TAGS = ("_mask", "-mask", "mask_", "mask-", "_seg", "_label")


@dataclass(frozen=True)
class MaskStatistics:
    values: tuple[int, ...]
    threshold: float
    average_person_fraction: float
    size_mismatches: int
    sample_count: int


class DatasetPairFinder:
    """Find matching image and mask files by their normalized file names."""

    @staticmethod
    def _normalized_stem(path: Path) -> str:
        stem = path.stem.lower()
        for tag in MASK_NAME_TAGS:
            stem = stem.replace(tag, "")
        return stem

    @classmethod
    def find_pairs(
        cls,
        dataset_root: Path,
        image_directory: Path | None = None,
        mask_directory: Path | None = None,
    ) -> list[ImageMaskPair]:
        if (image_directory is None) != (mask_directory is None):
            raise ValueError(
                "Both image_directory and mask_directory must be provided together."
            )
        if image_directory is not None and mask_directory is not None:
            image_root, mask_root = image_directory, mask_directory
            for directory in (image_root, mask_root):
                if not directory.is_dir():
                    raise NotADirectoryError(f"Dataset directory not found: {directory}")
            images = cls._files_by_stem(image_root.rglob("*"))
            masks = cls._files_by_stem(mask_root.rglob("*"))
        else:
            if not dataset_root.is_dir():
                raise NotADirectoryError(f"Dataset root not found: {dataset_root}")
            files = [
                path
                for path in dataset_root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            images = {}
            masks = {}
            for path in files:
                relative_path = path.relative_to(dataset_root).as_posix().lower()
                if "mask" in relative_path:
                    masks[cls._normalized_stem(path)] = path
                elif "collage" not in relative_path:
                    images[cls._normalized_stem(path)] = path

        pairs = [
            (images[key], masks[key])
            for key in sorted(images.keys() & masks.keys())
        ]
        if not pairs:
            raise ValueError(
                f"No matching image/mask files found under {dataset_root}. "
                "Check the folder names or pass both image and mask directories."
            )
        return pairs

    @classmethod
    def _files_by_stem(cls, paths: Iterable[Path]) -> dict[str, Path]:
        return {
            cls._normalized_stem(path): path
            for path in paths
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }


class MaskAnalyzer:
    """Measure mask encoding and check that image/mask pairs are plausible."""

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if mask is None:
            raise ValueError(f"Could not read mask image: {path}")
        return mask.max(axis=2)

    @classmethod
    def analyze(
        cls,
        pairs: Sequence[ImageMaskPair],
        sample_size: int = 100,
        seed: int = 42,
    ) -> MaskStatistics:
        if not pairs:
            raise ValueError("Cannot analyze an empty dataset.")
        if sample_size < 1:
            raise ValueError("sample_size must be positive.")

        sample = random.Random(seed).sample(pairs, min(sample_size, len(pairs)))
        masks = [cls._read_mask(mask_path) for _, mask_path in sample]
        unique_values = np.unique(np.concatenate([np.unique(mask) for mask in masks]))
        threshold = float(unique_values.max()) / 2
        foreground_fractions = [
            float(np.mean(mask > threshold)) for mask in masks
        ]
        average_person_fraction = float(np.mean(foreground_fractions))
        if unique_values.max() <= 0 or not 0.02 < average_person_fraction < 0.9:
            raise ValueError(
                "Masks appear empty or almost completely foreground. "
                "Check the image/mask pairing and mask encoding."
            )

        size_mismatches = 0
        for image_path, mask_path in sample:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            mask = cls._read_mask(mask_path)
            size_mismatches += image.shape[:2] != mask.shape[:2]
        if size_mismatches >= 0.5 * len(sample):
            raise ValueError(
                "At least half of the sampled image/mask sizes differ. "
                "Check that the correct folders were paired."
            )

        return MaskStatistics(
            values=tuple(int(value) for value in unique_values),
            threshold=threshold,
            average_person_fraction=average_person_fraction,
            size_mismatches=size_mismatches,
            sample_count=len(sample),
        )


class PersonDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load, resize, normalize, and binarize image/mask pairs."""

    def __init__(
        self,
        pairs: Sequence[ImageMaskPair],
        image_size: tuple[int, int],
        mask_threshold: float,
    ) -> None:
        if not pairs:
            raise ValueError("PersonDataset requires at least one image/mask pair.")
        if len(image_size) != 2 or min(image_size) < 1:
            raise ValueError("image_size must contain two positive dimensions.")
        if not 0 <= mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255.")
        self.pairs = list(pairs)
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        mask = MaskAnalyzer._read_mask(mask_path)

        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())

        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(
            (mask > self.mask_threshold).astype(np.float32)
        ).unsqueeze(0)
        return image_tensor, mask_tensor
