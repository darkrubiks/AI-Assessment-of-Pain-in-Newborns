import torch
from torch.utils.data import DataLoader, TensorDataset

from models.NCNN import NCNN
from utils.embeddings import extract_embeddings, find_nearest


def main() -> None:
    """Demonstrate embedding extraction and KNN search with NCNN."""
    # create a random dataset of 10 images sized 100x100
    images = torch.randn(10, 3, 100, 100)
    labels = torch.arange(10)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    model = NCNN()
    embeddings, labels = extract_embeddings(model, loader)

    # use the first embedding as query
    query = embeddings[0]
    idx, dist = find_nearest(query, embeddings, k=3)

    print("Closest labels:", labels[idx].tolist())
    print("Distances:", dist.tolist())


if __name__ == "__main__":
    main()
