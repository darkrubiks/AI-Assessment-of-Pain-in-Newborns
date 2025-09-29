"""Standalone ACE example using a small PyTorch CNN and synthetic data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from XAI.ACE import ACE, Concept


class TinyConvNet(nn.Module):
    """A compact CNN that works well on the synthetic shapes dataset."""

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


def generate_shapes_dataset(
    num_samples: int,
    *,
    seed: int,
    image_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create labelled RGB images containing squares on the left or right side."""

    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, image_size, image_size, 3), dtype=np.uint8)
    labels = np.zeros(num_samples, dtype=np.int64)

    x_grad = np.linspace(10.0, 80.0, image_size, dtype=np.float32)
    y_grad = np.linspace(60.0, 5.0, image_size, dtype=np.float32)

    for idx in range(num_samples):
        canvas = rng.uniform(0.0, 20.0, size=(image_size, image_size, 3)).astype(np.float32)
        canvas[..., 0] += x_grad[None, :]
        canvas[..., 1] += y_grad[:, None]
        canvas[..., 2] += x_grad[::-1][None, :]

        square_size = int(rng.integers(image_size // 6, image_size // 3))
        top = int(rng.integers(2, image_size - square_size - 2))
        left = int(rng.integers(2, image_size - square_size - 2))
        colour = rng.uniform(120.0, 240.0, size=3)
        canvas[top : top + square_size, left : left + square_size] = colour

        # Class 1 squares live on the right side of the canvas.
        square_center_x = left + square_size / 2.0
        labels[idx] = int(square_center_x >= image_size / 2.0)

        stripe_width = int(rng.integers(2, 4))
        stripe_colour = rng.uniform(40.0, 160.0, size=3)
        if rng.random() < 0.5:
            x = int(rng.integers(0, image_size - stripe_width))
            canvas[:, x : x + stripe_width] = stripe_colour
        else:
            y = int(rng.integers(0, image_size - stripe_width))
            canvas[y : y + stripe_width, :] = stripe_colour

        noise = rng.normal(0.0, 4.0, size=canvas.shape)
        noisy = np.clip(canvas + noise, 0.0, 255.0)
        images[idx] = noisy.astype(np.uint8)

    return images, labels


def train_model(
    model: TinyConvNet,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int = 32,
) -> None:
    dataset = TensorDataset(
        torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0,
        torch.from_numpy(labels),
    )
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


def evaluate_accuracy(model: TinyConvNet, images: np.ndarray, labels: np.ndarray, device: str) -> float:
    dataset = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    labels_tensor = torch.from_numpy(labels)
    device_obj = torch.device(device)

    model.eval()
    with torch.no_grad():
        logits = model(dataset.to(device_obj))
        predictions = logits.argmax(dim=1).cpu()
    accuracy = (predictions == labels_tensor).float().mean().item()
    return float(accuracy)


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
    parser.add_argument("--train-samples", type=int, default=240, help="Number of training images")
    parser.add_argument("--discovery-samples", type=int, default=24, help="Images used for concept discovery")
    parser.add_argument("--eval-samples", type=int, default=24, help="Images used for TCAV scoring")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs for the tiny CNN")
    parser.add_argument("--num-concepts", type=int, default=5, help="Number of ACE clusters")
    parser.add_argument("--output", type=Path, default=Path("ace_pytorch_outputs"), help="Directory for artefacts")
    args = parser.parse_args()

    print("Preparing synthetic datasets...")
    train_images, train_labels = generate_shapes_dataset(args.train_samples, seed=0)
    discovery_images, _ = generate_shapes_dataset(args.discovery_samples, seed=1)
    evaluation_images, evaluation_labels = generate_shapes_dataset(args.eval_samples, seed=2)

    model = TinyConvNet(num_classes=2)
    print("Training the CNN on the synthetic task...")
    train_model(model, train_images, train_labels, device=args.device, epochs=args.epochs)

    train_acc = evaluate_accuracy(model, train_images, train_labels, device=args.device)
    eval_acc = evaluate_accuracy(model, evaluation_images, evaluation_labels, device=args.device)
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Evaluation accuracy: {eval_acc:.3f}")

    ace = ACE(model, target_layer=model.conv3, device=args.device, batch_size=16)

    discovery_list = [img for img in discovery_images]
    print("Discovering concepts...")
    result = ace.discover_concepts(
        discovery_list,
        n_segments=12,
        compactness=8.0,
        sigma=1.0,
        n_concepts=args.num_concepts,
        random_state=0,
    )
    print(f"Discovered {len(result.concepts)} concepts")

    print("Training CAVs...")
    ace.train_cavs(result, random_sample_size=80, random_state=0)

    evaluation_list = [img for img in evaluation_images]
    print("Scoring concepts with TCAV...")
    scores = ace.score_concepts(result, evaluation_list, class_index=1)
    for concept_id, score in sorted(scores.items()):
        print(f"Concept {concept_id}: TCAV score = {score:.3f}")

    plot_dir = args.output / "concept_plots"
    plot_concepts(result.concepts.values(), discovery_list, plot_dir)
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
