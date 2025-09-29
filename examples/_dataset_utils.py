"""Helper utilities for loading face image datasets in ACE examples."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ImageTransform = Callable[[Image.Image], torch.Tensor]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_image_paths(dataset_dir: Path) -> List[Path]:
    """Return all image paths under ``dataset_dir`` sorted alphabetically."""

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path '{dataset_dir}' is not a directory")

    paths = [path for path in dataset_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    paths.sort()
    if not paths:
        raise ValueError(
            f"No images with extensions {sorted(SUPPORTED_EXTENSIONS)} were found in '{dataset_dir}'"
        )
    return paths


def infer_binary_label(path: Path) -> int:
    """Infer a binary pain label from the filename following the project's convention."""

    token = path.stem.split("_")[-1].lower()
    if "nopain" in token:
        return 0
    if "pain" in token:
        return 1
    raise ValueError(f"Could not infer a 'pain' or 'nopain' label from '{path.name}'")


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``CHW`` tensor in ``[0, 1]`` to an ``HWC`` uint8 NumPy array."""

    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.clip(array * 255.0, 0.0, 255.0).round().astype(np.uint8)


def load_numpy_images(paths: Sequence[Path], transform: ImageTransform) -> List[np.ndarray]:
    """Load images from ``paths`` applying ``transform`` and returning uint8 arrays."""

    images: List[np.ndarray] = []
    for path in paths:
        with Image.open(path) as pil_image:
            tensor = transform(pil_image.convert("RGB"))
        images.append(tensor_to_uint8_image(tensor))
    return images


def select_image_subsets(
    image_paths: Sequence[Path], counts: Sequence[int], *, seed: int
) -> List[List[Path]]:
    """Split ``image_paths`` into non-overlapping subsets according to ``counts``."""

    if any(count < 0 for count in counts):
        raise ValueError("Sample counts must be non-negative")

    total_required = int(sum(counts))
    if total_required > len(image_paths):
        raise ValueError(
            f"Requested {total_required} images across all splits but dataset contains only {len(image_paths)} files. "
            "Reduce the requested sample counts."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(image_paths))

    subsets: List[List[Path]] = []
    offset = 0
    for count in counts:
        indices = permutation[offset : offset + count]
        subsets.append([image_paths[int(idx)] for idx in indices])
        offset += count
    return subsets


class FaceImageDataset(Dataset):
    """Torch ``Dataset`` returning tensors and binary labels for face images."""

    def __init__(self, image_paths: Iterable[Path], transform: ImageTransform) -> None:
        self.image_paths: List[Path] = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.image_paths)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path = self.image_paths[idx]
        with Image.open(path) as pil_image:
            tensor = self.transform(pil_image.convert("RGB"))
        label = infer_binary_label(path)
        return tensor, torch.tensor(label, dtype=torch.long)

