"""ACE example that trains a tiny CNN on face images stored on disk."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import numpy as np

from dataloaders.presets import PresetTransform
from examples._dataset_utils import (
    FaceImageDataset,
    collect_image_paths,
    load_numpy_images,
    select_image_subsets,
)
from XAI.ACE import ACE, Concept


class TinyConvNet(nn.Module):
    """A compact CNN suited for small binary pain classification experiments."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_model(
    model: TinyConvNet,
    dataset: FaceImageDataset,
    *,
    device: str,
    epochs: int,
    batch_size: int = 32,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device_obj = torch.device(device)
    model.to(device_obj)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device_obj)
            batch_labels = batch_labels.to(device_obj)

            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * batch_images.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - training loss: {avg_loss:.4f}")

    model.eval()


def evaluate_accuracy(model: TinyConvNet, dataset: FaceImageDataset, device: str) -> float:
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device_obj = torch.device(device)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            logits = model(images.to(device_obj))
            predictions = logits.argmax(dim=1).cpu()
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    if total == 0:
        return 0.0
    return correct / total


def plot_concepts(
    concepts: Iterable[Concept],
    discovery_images: Sequence[np.ndarray],
    output_dir: Path,
    *,
    max_examples: int = 4,
    overlay_alpha: float = 0.6,
) -> None:
    """Render per-concept PNG grids highlighting the most central patches."""

    output_dir.mkdir(parents=True, exist_ok=True)
    red_overlay = np.array([255.0, 0.0, 0.0], dtype=np.float32)

    for concept in concepts:
        if not concept.patches:
            continue

        top_patches = concept.patches[:max_examples]
        rows = len(top_patches)
        fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows), squeeze=False)

        for row, patch in enumerate(top_patches):
            original = discovery_images[patch.image_index]
            mask = patch.mask.astype(bool)

            highlight = original.astype(np.float32).copy()
            highlight[mask] = (
                highlight[mask] * (1.0 - overlay_alpha) + red_overlay * overlay_alpha
            )
            highlight = np.clip(highlight, 0, 255).astype(np.uint8)

            axes[row, 0].imshow(original)
            axes[row, 0].set_title(f"Image {patch.image_index} – patch {patch.patch_index}")
            axes[row, 1].imshow(highlight)
            axes[row, 1].set_title("Highlighted concept region")

            for ax in axes[row]:
                ax.axis("off")

        fig.suptitle(f"Concept {concept.concept_id}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_dir / f"concept_{concept.concept_id}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end ACE demo with a PyTorch CNN")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Datasets") / "DatasetsFaces" / "Images",
        help="Directory containing the face images (expects *_pain/*.jpg naming)",
    )
    parser.add_argument("--train-samples", type=int, default=192, help="Number of training images")
    parser.add_argument("--discovery-samples", type=int, default=32, help="Images used for concept discovery")
    parser.add_argument("--eval-samples", type=int, default=32, help="Images used for TCAV scoring")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs for the tiny CNN")
    parser.add_argument("--num-concepts", type=int, default=5, help="Number of ACE clusters")
    parser.add_argument("--output", type=Path, default=Path("ace_pytorch_outputs"), help="Directory for artefacts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset splits")
    args = parser.parse_args()

    transform = PresetTransform("NCNN").transform
    dataset_dir = args.dataset_dir
    image_paths = collect_image_paths(dataset_dir)

    train_paths, discovery_paths, evaluation_paths = select_image_subsets(
        image_paths,
        [args.train_samples, args.discovery_samples, args.eval_samples],
        seed=args.seed,
    )

    print(
        f"Loaded {len(image_paths)} images from {dataset_dir}. "
        f"Using {len(train_paths)} for training, {len(discovery_paths)} for concept discovery, "
        f"and {len(evaluation_paths)} for TCAV scoring."
    )

    train_dataset = FaceImageDataset(train_paths, transform)
    evaluation_dataset = FaceImageDataset(evaluation_paths, transform)
    discovery_images = load_numpy_images(discovery_paths, transform)
    evaluation_images = load_numpy_images(evaluation_paths, transform)

    model = TinyConvNet(num_classes=2)
    print("Training the CNN on the neonatal pain dataset subset...")
    train_model(model, train_dataset, device=args.device, epochs=args.epochs)

    train_acc = evaluate_accuracy(model, train_dataset, device=args.device)
    eval_acc = evaluate_accuracy(model, evaluation_dataset, device=args.device)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Evaluation accuracy: {eval_acc:.3f}")

    ace = ACE(model, target_layer=model.conv3, device=args.device, batch_size=16)

    print("Discovering concepts...")
    result = ace.discover_concepts(
        discovery_images,
        n_segments=12,
        compactness=8.0,
        sigma=1.0,
        n_concepts=args.num_concepts,
        random_state=0,
    )
    print(f"Discovered {len(result.concepts)} concepts")

    print("Training CAVs...")
    ace.train_cavs(result, random_sample_size=80, random_state=0)

    print("Scoring concepts with TCAV...")
    scores = ace.score_concepts(result, evaluation_images, class_index=1)
    for concept_id, score in sorted(scores.items()):
        print(f"Concept {concept_id}: TCAV score = {score:.3f}")

    plot_dir = args.output / "concept_plots"
    plot_concepts(result.concepts.values(), discovery_images, plot_dir)
    print(f"Saved concept visualisations to {plot_dir}")

    mask_dir = args.output / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for concept in result.concepts.values():
        for patch in concept.patches[:3]:
            mask_path = mask_dir / f"concept{concept.concept_id}_img{patch.image_index}_patch{patch.patch_index}.npy"
            np.save(mask_path, patch.mask)

    print(f"Stored example masks in {mask_dir}")
    ace.close()


if __name__ == "__main__":
    main()
