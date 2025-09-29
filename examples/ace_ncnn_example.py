"""Run the ACE concept discovery pipeline on the NCNN model using real images."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import math

import torch
from matplotlib import pyplot as plt

import numpy as np

from dataloaders.presets import PresetTransform
from examples._dataset_utils import collect_image_paths, load_numpy_images, select_image_subsets
from models.NCNN import NCNN
from XAI.ACE import ACE, Concept


class NCNNForACE(NCNN):
    """Wrap ``NCNN`` so that its output is always 2-D for TCAV scoring."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = super().forward(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        return logits


def _save_masks(concepts: Iterable[Concept], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        for patch in concept.patches[:3]:  # only store a few masks per concept
            mask_path = output_dir / f"concept{concept.concept_id}_img{patch.image_index}_patch{patch.patch_index}.npy"
            np.save(mask_path, patch.mask)


def _plot_concepts(
    concepts: Iterable[Concept],
    discovery_images: List[np.ndarray],
    output_dir: Path,
    *,
    max_examples: int = 9,
) -> None:
    """Render PNG mosaics showing the raw patches for each concept."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for concept in concepts:
        if not concept.patches:
            continue

        top_patches = concept.patches[:max_examples]
        n_patches = len(top_patches)
        cols = min(3, n_patches)
        rows = math.ceil(n_patches / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_2d(axes)

        for idx, patch in enumerate(top_patches):
            row, col = divmod(idx, cols)
            original = discovery_images[patch.image_index]
            mask = patch.mask.astype(bool)
            patch_only = np.zeros_like(original)
            patch_only[mask] = original[mask]

            ax = axes[row, col]
            ax.imshow(patch_only)
            ax.set_title(f"Img {patch.image_index} – patch {patch.patch_index}")
            ax.axis("off")

        for idx in range(n_patches, rows * cols):
            row, col = divmod(idx, cols)
            axes[row, col].axis("off")

        fig.suptitle(f"Concept {concept.concept_id}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_dir / f"concept_{concept.concept_id}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACE demo on the NCNN model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Datasets") / "DatasetsFaces" / "Images",
        help="Directory containing the face images (expects *_pain/*.jpg naming)",
    )
    parser.add_argument("--discovery-samples", type=int, default=24, help="Number of images for concept discovery")
    parser.add_argument("--eval-samples", type=int, default=12, help="Number of images for TCAV scoring")
    parser.add_argument("--num-concepts", type=int, default=3, help="Number of ACE clusters")
    parser.add_argument("--output", type=Path, default=Path("ace_outputs"), help="Directory for saved masks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset sampling")
    args = parser.parse_args()

    transform = PresetTransform("NCNN").transform
    image_paths = collect_image_paths(args.dataset_dir)
    discovery_paths, evaluation_paths = select_image_subsets(
        image_paths,
        [args.discovery_samples, args.eval_samples],
        seed=args.seed,
    )

    print(
        f"Loaded {len(image_paths)} images from {args.dataset_dir}. "
        f"Using {len(discovery_paths)} for discovery and {len(evaluation_paths)} for TCAV scoring."
    )

    discovery_images = load_numpy_images(discovery_paths, transform)
    evaluation_images = load_numpy_images(evaluation_paths, transform)

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

    mask_dir = args.output / "masks"
    _save_masks(result.concepts.values(), mask_dir)
    print(f"Saved example masks into {mask_dir}")

    plot_dir = args.output / "plots"
    _plot_concepts(result.concepts.values(), discovery_images, plot_dir)
    print(f"Rendered concept visualisations into {plot_dir}")

    ace.close()


if __name__ == "__main__":
    main()
