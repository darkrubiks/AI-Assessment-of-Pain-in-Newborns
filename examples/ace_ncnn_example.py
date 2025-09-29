"""Run the ACE concept discovery pipeline on the NCNN model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from models.NCNN import NCNN
from XAI.ACE import ACE, Concept


class NCNNForACE(NCNN):
    """Wrap ``NCNN`` so that its output is always 2-D for TCAV scoring."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = super().forward(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        return logits


def _make_synthetic_image(seed: int, size: int = 100) -> np.ndarray:
    """Create a simple RGB image with geometric patterns for ACE to segment."""

    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.float32)

    # Background gradient
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    image[..., 0] = xs[None, :]
    image[..., 1] = xs[:, None]

    # Add coloured rectangles so SLIC can find diverse patches
    for _ in range(5):
        h = rng.integers(size // 8, size // 3)
        w = rng.integers(size // 8, size // 3)
        top = rng.integers(0, size - h)
        left = rng.integers(0, size - w)
        colour = rng.uniform(0.2, 0.9, size=3).astype(np.float32)
        image[top : top + h, left : left + w] = colour

    # Convert to uint8 style range [0, 255]
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _generate_dataset(num_images: int, size: int = 100) -> List[np.ndarray]:
    return [_make_synthetic_image(seed=i, size=size) for i in range(num_images)]


def _save_masks(concepts: Iterable[Concept], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        for patch in concept.patches[:3]:  # only store a few masks per concept
            mask_path = output_dir / f"concept{concept.concept_id}_img{patch.image_index}_patch{patch.patch_index}.npy"
            np.save(mask_path, patch.mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACE demo on the NCNN model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-images", type=int, default=12, help="Number of discovery images")
    parser.add_argument("--num-concepts", type=int, default=3, help="Number of ACE clusters")
    parser.add_argument("--output", type=Path, default=Path("ace_outputs"), help="Directory for saved masks")
    args = parser.parse_args()

    discovery_images = _generate_dataset(args.num_images)
    evaluation_images = _generate_dataset(4)

    model = NCNNForACE(num_classes=1)
    ace = ACE(model, target_layer=model.merge_branch, device=args.device, batch_size=8)

    print("Discovering concepts...")
    result = ace.discover_concepts(
        discovery_images,
        n_segments=10,
        compactness=8.0,
        n_concepts=args.num_concepts,
        random_state=0,
    )
    print(f"Discovered {len(result.concepts)} concepts")

    print("Training CAVs...")
    ace.train_cavs(result, random_sample_size=50, random_state=0)

    print("Scoring concepts with TCAV...")
    scores = ace.score_concepts(result, evaluation_images, class_index=0)
    for concept_id, score in sorted(scores.items()):
        print(f"Concept {concept_id}: TCAV score = {score:.3f}")

    _save_masks(result.concepts.values(), args.output)
    print(f"Saved example masks into {args.output}")

    ace.close()


if __name__ == "__main__":
    main()
