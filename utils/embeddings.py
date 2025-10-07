import torch
from typing import Iterable, Tuple, Optional

@torch.no_grad()
def extract_embeddings(model: torch.nn.Module,
                       dataloader: Iterable,
                       device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract embeddings for all samples in a dataloader.

    Parameters
    ----------
    model: torch.nn.Module
        Model implementing a ``get_embedding`` method.
    dataloader: Iterable
        Iterable yielding ``(images, label)`` or ``images``.
    device: torch.device, optional
        Device to perform computation. If ``None`` uses the model's device.

    Returns
    -------
    embeddings: torch.Tensor
        Tensor of shape ``[N, D]`` with all embeddings.
    labels: torch.Tensor or None
        Tensor with labels if provided by the dataloader.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    all_embs = []
    all_labels = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            images, labels = batch
            all_labels.append(torch.as_tensor(labels))
        else:
            images = batch
        images = images.to(device)
        emb = model.get_embedding(images).detach().cpu()
        all_embs.append(emb)
    embeddings = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else None
    return embeddings, labels


def find_nearest(query_embedding: torch.Tensor,
                 embeddings: torch.Tensor,
                 k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find ``k`` nearest embeddings to ``query_embedding`` using L2 distance."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)
    diff = embeddings - query_embedding
    distances = diff.norm(dim=1)
    dist, idx = torch.topk(distances, k, largest=False)
    return idx, dist


