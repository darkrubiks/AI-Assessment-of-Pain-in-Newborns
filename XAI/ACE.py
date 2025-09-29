"""PyTorch implementation of the Automated Concept-based Explanations (ACE) method."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier
from skimage.segmentation import slic
from torch.nn import functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]


def _default_preprocess(image: np.ndarray) -> torch.Tensor:
    """Convert a numpy image in HWC format to a float tensor in CHW format."""
    if image.ndim != 3:
        raise ValueError(f"Expected an image with 3 dimensions (HWC), got {image.shape}")
    tensor = torch.from_numpy(image.astype(np.float32))
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    # convert to CHW
    tensor = tensor.permute(2, 0, 1)
    return tensor


@dataclass
class ConceptPatch:
    """Metadata describing a single super-pixel patch used for concept discovery."""

    image_index: int
    patch_index: int
    global_index: int
    mask: np.ndarray
    activation: np.ndarray
    distance_to_center: float


@dataclass
class Concept:
    """Represents a discovered concept and optional attribution metrics."""

    concept_id: int
    patches: List[ConceptPatch]
    activation_vectors: np.ndarray
    cav: Optional[np.ndarray] = None
    tcav_score: Optional[float] = None


@dataclass
class ACEResult:
    """Stores the outcome of the concept discovery stage."""

    concepts: Dict[int, Concept]
    assignments: np.ndarray
    activations: np.ndarray
    metadata: List[ConceptPatch]
    kmeans: KMeans


class ACE:
    """PyTorch implementation of the Automated Concept-based Explanations (ACE).

    The method follows the pipeline described in the original paper:

    1. Generate super-pixels for the images of a target class.
    2. Extract activations from a chosen internal layer for each super-pixel patch.
    3. Cluster the activations to obtain coherent concepts.
    4. Optionally train Concept Activation Vectors (CAVs) and compute TCAV scores.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` that will be analysed. The model is expected to
        output class logits.
    target_layer:
        The internal module whose activations should be used for concept discovery.
        It can be either a string (name of the layer obtained via
        ``model.named_modules()``) or the actual module instance.
    device:
        Device on which the computations will run (``"cpu"`` or ``"cuda"``).
    preprocess:
        Callable applied to each numpy image before being fed into the network.
        The callable must return a tensor in ``CHW`` format.
    batch_size:
        Number of patches processed in a single forward pass when extracting
        activations.
    activation_aggregator:
        Optional callable transforming the raw feature map returned by
        ``target_layer``. When ``None``, global average pooling is applied.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Union[str, torch.nn.Module],
        device: str = "cpu",
        preprocess: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        batch_size: int = 16,
        activation_aggregator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self._disable_inplace_activations()

        self.preprocess = preprocess or _default_preprocess
        self.batch_size = batch_size
        self.activation_aggregator = activation_aggregator

        self._activation: Optional[torch.Tensor] = None
        self._activation_grad: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks(target_layer)

    def _disable_inplace_activations(self) -> None:
        """Set ``inplace`` to ``False`` on modules that support it.

        Many pretrained CNNs (e.g., VGG variants) rely on in-place activation
        functions such as :class:`torch.nn.ReLU`. When ACE registers hooks on
        intermediate layers, these in-place operations can overwrite the
        tensors captured by the hooks, leading to autograd complaints about
        views being modified in-place. To avoid this, the helper scans the
        model and disables the in-place behaviour whenever possible.
        """

        for module in self.model.modules():
            if hasattr(module, "inplace") and getattr(module, "inplace"):
                try:
                    module.inplace = False  # type: ignore[assignment]
                except AttributeError:
                    # Some modules expose ``inplace`` as a read-only property;
                    # in that case we simply skip them.
                    continue


    # ------------------------------------------------------------------
    # Hook utilities
    # ------------------------------------------------------------------
    def _register_hooks(self, target_layer: Union[str, torch.nn.Module]) -> None:
        if isinstance(target_layer, str):
            modules = dict(self.model.named_modules())
            if target_layer not in modules:
                raise ValueError(f"Layer '{target_layer}' not found in the model")
            module = modules[target_layer]
        else:
            module = target_layer

        def forward_hook(_module: torch.nn.Module, _input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._activation = output.detach().clone()

        def backward_hook(
            _module: torch.nn.Module, _input: Tuple[torch.Tensor, ...], output_grad: Tuple[torch.Tensor, ...]
        ) -> None:
            grad = output_grad[0]
            self._activation_grad = grad.detach().clone()

        self._handles.append(module.register_forward_hook(forward_hook))
        self._handles.append(module.register_full_backward_hook(backward_hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Image and patch processing utilities
    # ------------------------------------------------------------------
    def _prepare_tensor(self, image: ArrayLike) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.clone().detach()
            if tensor.ndim == 3:
                pass
            elif tensor.ndim == 4:
                if tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                else:
                    raise ValueError("Only single-image tensors are supported")
            else:
                raise ValueError(f"Unsupported tensor shape {tuple(tensor.shape)}")
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            return tensor
        if not isinstance(image, np.ndarray):
            raise TypeError("Images must be provided as numpy arrays or torch tensors")
        return self.preprocess(image)

    def _segment_image(
        self,
        image: np.ndarray,
        n_segments: int,
        compactness: float,
        sigma: float,
    ) -> np.ndarray:
        segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=0)
        return segments

    def _extract_patches(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        min_area_ratio: float,
        max_area_ratio: float,
    ) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        unique_segments = np.unique(segments)
        h, w = segments.shape
        for segment_id in unique_segments:
            mask = segments == segment_id
            area_ratio = float(mask.sum()) / float(h * w)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            patches.append(mask.astype(np.float32))
        return patches

    def _create_patch_tensors(
        self,
        image_tensor: torch.Tensor,
        masks: Sequence[np.ndarray],
    ) -> torch.Tensor:
        tensors = []
        image_tensor = image_tensor.to(self.device)
        for mask in masks:
            mask_tensor = torch.from_numpy(mask).to(self.device)
            if mask_tensor.ndim != 2:
                raise ValueError("Masks must be 2-D arrays")
            mask_tensor = mask_tensor.unsqueeze(0)
            patch = image_tensor * mask_tensor
            tensors.append(patch)
        if not tensors:
            return torch.empty(0, *image_tensor.shape, device=self.device)
        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------
    def _aggregate_activation(self, activation: torch.Tensor) -> torch.Tensor:
        if self.activation_aggregator is not None:
            return self.activation_aggregator(activation)
        if activation.ndim <= 2:
            return activation
        batch = activation.shape[0]
        return activation.view(batch, activation.shape[1], -1).mean(dim=-1)

    def _collect_activations(self, tensors: torch.Tensor) -> np.ndarray:
        if tensors.numel() == 0:
            return np.empty((0,))
        activations: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, tensors.shape[0], self.batch_size):
                batch = tensors[start : start + self.batch_size]
                batch = batch.to(self.device)
                self._activation = None
                self.model(batch)
                if self._activation is None:
                    raise RuntimeError("Forward hook did not capture activations")
                activation = self._aggregate_activation(self._activation)
                activations.append(activation.cpu().numpy())
        return np.concatenate(activations, axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover_concepts(
        self,
        images: Sequence[np.ndarray],
        *,
        n_segments: int = 15,
        compactness: float = 10.0,
        sigma: float = 1.0,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.5,
        n_concepts: int = 10,
        random_state: int = 0,
    ) -> ACEResult:
        """Discover concepts on the provided images.

        Parameters
        ----------
        images:
            Sequence of images (numpy arrays in HWC format).
        n_segments:
            Number of super-pixels generated by ``slic`` for each image.
        compactness:
            ``slic`` compactness parameter.
        sigma:
            ``slic`` sigma parameter.
        min_area_ratio / max_area_ratio:
            Filters super-pixels that are too small or too large.
        n_concepts:
            Number of clusters (concepts) to discover.
        random_state:
            Random seed for reproducible clustering.
        """

        all_patch_masks: List[np.ndarray] = []
        patch_image_indices: List[int] = []
        per_image_patch_ids: List[int] = []
        all_tensors: List[torch.Tensor] = []

        target_spatial_size: Optional[Tuple[int, int]] = None

        for img_idx, image in enumerate(images):
            if not isinstance(image, np.ndarray):
                raise TypeError("Images must be provided as numpy arrays when discovering concepts")
            segments = self._segment_image(image, n_segments, compactness, sigma)
            masks = self._extract_patches(image, segments, min_area_ratio, max_area_ratio)
            if not masks:
                continue
            image_tensor = self._prepare_tensor(image)
            patch_tensors = self._create_patch_tensors(image_tensor, masks)
            if patch_tensors.numel() == 0:
                continue
            if target_spatial_size is None:
                target_spatial_size = patch_tensors.shape[-2:]
            elif patch_tensors.shape[-2:] != target_spatial_size:
                patch_tensors = F.interpolate(
                    patch_tensors,
                    size=target_spatial_size,
                    mode="bilinear",
                    align_corners=False,
                )
            all_patch_masks.extend(masks)
            patch_image_indices.extend([img_idx] * len(masks))
            per_image_patch_ids.extend(list(range(len(masks))))
            all_tensors.append(patch_tensors)

        if not all_tensors:
            raise ValueError("No valid patches found for the provided images")

        patch_tensor = torch.cat(all_tensors, dim=0)
        activations = self._collect_activations(patch_tensor)
        if activations.ndim == 1:
            activations = activations[:, None]

        kmeans = KMeans(n_clusters=n_concepts, random_state=random_state)
        assignments = kmeans.fit_predict(activations)

        metadata: List[ConceptPatch] = []
        concepts: Dict[int, List[ConceptPatch]] = {cluster_id: [] for cluster_id in range(n_concepts)}

        for global_index, (cluster_id, mask, act, img_idx, patch_idx) in enumerate(
            zip(assignments, all_patch_masks, activations, patch_image_indices, per_image_patch_ids)
        ):
            center = kmeans.cluster_centers_[cluster_id]
            distance = float(np.linalg.norm(act - center))
            patch_meta = ConceptPatch(
                image_index=img_idx,
                patch_index=patch_idx,
                global_index=global_index,
                mask=mask,
                activation=act,
                distance_to_center=distance,
            )
            metadata.append(patch_meta)
            concepts[cluster_id].append(patch_meta)

        concept_objects: Dict[int, Concept] = {}
        for concept_id, patches in concepts.items():
            if not patches:
                continue
            activation_vectors = np.stack([p.activation for p in patches], axis=0)
            sorted_patches = sorted(patches, key=lambda p: p.distance_to_center)
            concept_objects[concept_id] = Concept(
                concept_id=concept_id,
                patches=sorted_patches,
                activation_vectors=activation_vectors,
            )

        return ACEResult(
            concepts=concept_objects,
            assignments=assignments,
            activations=activations,
            metadata=metadata,
            kmeans=kmeans,
        )

    def train_cavs(
        self,
        result: ACEResult,
        *,
        random_sample_size: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        """Train Concept Activation Vectors (CAVs) for every discovered concept.

        Parameters
        ----------
        result:
            The output of :meth:`discover_concepts`.
        random_sample_size:
            Number of negative examples sampled from the remaining patches. When
            ``None`` all non-concept patches are used.
        random_state:
            Random seed for the logistic regression classifier.
        """

        rng = np.random.default_rng(random_state)
        all_indices = np.arange(len(result.metadata))
        all_activations = result.activations

        for concept_id, concept in result.concepts.items():
            positive_indices = np.array([patch.global_index for patch in concept.patches], dtype=int)
            negative_indices = np.setdiff1d(all_indices, positive_indices, assume_unique=False)
            if random_sample_size is not None and random_sample_size < len(negative_indices):
                negative_indices = rng.choice(negative_indices, size=random_sample_size, replace=False)

            positives = all_activations[positive_indices]
            negatives = all_activations[negative_indices]
            if len(negatives) == 0:
                raise ValueError(
                    "No negative examples available to train a CAV. Try increasing the number of discovered concepts."
                )

            X = np.concatenate([positives, negatives], axis=0)
            y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))], axis=0)

            if len(np.unique(y)) < 2:
                raise ValueError(f"Insufficient data to train a CAV for concept {concept_id}")

            classifier = SGDClassifier(
                alpha=0.01,
                max_iter=1000,
                tol=1e-3
            )
            classifier.fit(X, y)
            concept.cav = classifier.coef_.reshape(-1)

    def score_concepts(
        self,
        result: ACEResult,
        images: Sequence[ArrayLike],
        class_index: int,
    ) -> Dict[int, float]:
        """Compute TCAV scores for a set of images.

        Parameters
        ----------
        result:
            The concept discovery output with trained CAVs.
        images:
            Input images used to evaluate concept importance.
        class_index:
            Target class index whose logit will be differentiated.
        """

        for concept_id, concept in result.concepts.items():
            if concept.cav is None:
                raise ValueError(
                    "Concept %d does not have a trained CAV. Call `train_cavs` before scoring." % concept_id
                )

        gradients: List[np.ndarray] = []

        for image in images:
            tensor = self._prepare_tensor(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            tensor.requires_grad_(True)
            self.model.zero_grad()
            self._activation = None
            self._activation_grad = None
            output = self.model(tensor)
            logit = output[:, class_index].sum()
            logit.backward()
            if self._activation is None or self._activation_grad is None:
                raise RuntimeError("Hooks did not capture activations and gradients")
            activation_grad = self._aggregate_activation(self._activation_grad)
            grad_vector = activation_grad.detach().cpu().numpy().reshape(-1)
            gradients.append(grad_vector)

        gradients = np.stack(gradients, axis=0)

        scores: Dict[int, float] = {}
        for concept_id, concept in result.concepts.items():
            cav = concept.cav.reshape(-1)
            directional_derivatives = gradients.dot(cav)
            tcav_score = float((directional_derivatives > 0).mean())
            concept.tcav_score = tcav_score
            scores[concept_id] = tcav_score
        return scores


__all__ = [
    "ACE",
    "ACEResult",
    "Concept",
    "ConceptPatch",
]
