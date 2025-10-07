import torch
from typing import Iterable, Tuple, Optional
from tqdm import tqdm
import numpy as np

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
    paths: list[str]
        List with the file paths associated with each embedding.
    """
    model.eval()
    model.to(device)
    all_embs = []
    all_labels = []
    all_paths = []

    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        paths = batch['path']

        all_labels.append(torch.as_tensor(labels))
        all_paths.extend(paths)

        images = images.to(device)
        emb = model.get_embedding(images).detach().cpu()
        all_embs.append(emb)

    embeddings = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else None

    result = {
        "embeddings": embeddings.cpu().numpy(),
        "labels": labels.cpu().numpy(),
        "paths": np.asarray(all_paths, dtype=object),
    }
    
    return result


import numpy as np
from typing import Dict, Tuple

def find_nearest(
    query_embedding,
    embedding_store,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find ``k`` nearest embeddings to ``query_embedding`` using L2 distance.

    Parameters
    ----------
    query_embedding : np.ndarray | torch.Tensor
        The embedding to query, shape ``[D]`` or ``[1, D]``.
    embedding_store : dict[str, np.ndarray]
        Dictionary containing keys ``embeddings`` (``[N, D]``) and ``paths`` (``[N]``).
    k : int, optional
        Number of nearest neighbours to return.

    Returns
    -------
    indices : np.ndarray
        Indices of the nearest embeddings within ``embedding_store["embeddings"]``.
    distances : np.ndarray
        L2 distances corresponding to each neighbour.
    paths : np.ndarray
        The file paths (dtype ``object``) for each neighbour.
    """
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.detach().cpu().numpy()
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[np.newaxis, :]

    embeddings = embedding_store["embeddings"]
    paths = embedding_store.get("paths", np.array([], dtype=object))

    diff = embeddings - query_embedding
    distances = np.linalg.norm(diff, axis=1)

    k = min(k, embeddings.shape[0])
    nearest_idx = np.argpartition(distances, k - 1)[:k]
    nearest_idx = nearest_idx[np.argsort(distances[nearest_idx])]

    return nearest_idx, distances[nearest_idx], paths[nearest_idx]


