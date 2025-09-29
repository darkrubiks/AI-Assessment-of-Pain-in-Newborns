"""Run the ACE concept discovery pipeline on the VGGFace model."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from dataloaders.presets import PresetTransform
from examples._dataset_utils import collect_image_paths, load_numpy_images, select_image_subsets
from models.VGGFace import VGGFace
from XAI.ACE import ACE, Concept


def _build_visual_transform() -> transforms.Compose:
    """Return a resize + tensor transform without VGGFace normalisation."""

    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    )


def _save_masks(concepts: Iterable[Concept], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        for patch in concept.patches[:3]:
            mask_path = output_dir / f"concept{concept.concept_id}_img{patch.image_index}_patch{patch.patch_index}.npy"
            np.save(mask_path, patch.mask)


def _plot_concepts(
    concepts: Iterable[Concept],
    discovery_images: List[np.ndarray],
    output_dir: Path,
    *,
    max_examples: int = 9,
) -> None:
    """Render PNG mosaics showing only the raw patch pixels for each concept."""

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


def _build_preprocess(preset: PresetTransform):
    """Create a preprocessing callable compatible with :class:`ACE`."""

    transform = preset.transform

    def preprocess(image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image_np = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image_np = image
        pil_image = Image.fromarray(image_np)
        return transform(pil_image.convert("RGB"))

    return preprocess


def _resolve_conv53_layer(model: VGGFace) -> torch.nn.Module:
    """Return the ``conv5_3`` layer from the wrapped VGGFace backbone."""

    for name, module in model.VGGFace.named_modules():
        if name.endswith("conv5_3"):
            return module
    raise RuntimeError("conv5_3 layer could not be located in the VGGFace backbone")


class VGGFaceForACE(VGGFace):
    """Wrapper ensuring the forward pass returns 2-D logits for TCAV scoring."""

    def __init__(self, weights_path: Path | None = None) -> None:
        super().__init__(weights_path=weights_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.VGGFace(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim == 2:
            return logits
        return logits.view(logits.size(0), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACE demo on the VGGFace model")
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
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional path to VGGFace weights (defaults to models/weights/VGG_face_original_model.pt)",
    )
    parser.add_argument("--output", type=Path, default=Path("ace_vggface_outputs"), help="Directory for saved artefacts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset sampling")
    args = parser.parse_args()

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

    visual_transform = _build_visual_transform()
    discovery_images = load_numpy_images(discovery_paths, visual_transform)
    evaluation_images = load_numpy_images(evaluation_paths, visual_transform)

    model = VGGFaceForACE(weights_path=args.weights)
    conv5_3 = _resolve_conv53_layer(model)

    preset = PresetTransform("VGGFace")
    preprocess = _build_preprocess(preset)

    ace = ACE(model, target_layer=conv5_3, device=args.device, batch_size=8, preprocess=preprocess)

    print("Discovering concepts...")
    result = ace.discover_concepts(
        discovery_images,
        n_segments=15,
        compactness=10.0,
        sigma=1.0,
        n_concepts=args.num_concepts,
        random_state=0,
    )
    print(f"Discovered {len(result.concepts)} concepts")

    print("Training CAVs...")
    ace.train_cavs(result, random_sample_size=80, random_state=0)

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
