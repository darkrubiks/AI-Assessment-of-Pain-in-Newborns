# NCNN.py

# Author: Leonardo Antunes Ferreira
# Date: 05/01/2025
#
# Based on:
# G. Zamzmi et al., "Pain Assessment From Facial Expression: Neonatal Convolutional Neural Network (N-CNN)",
# IEEE IJCNN 2019, doi:10.1109/IJCNN.2019.8851879

import torch
import torch.nn as nn


class NCNN(nn.Module):
    """NCNN with a clear separation between feature-extractor and classifier,
       but contained in a single class."""
    def __init__(self, num_classes: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        # --- Feature extractor ---
        # Left branch
        self.left_branch = nn.Sequential(
            nn.MaxPool2d(10, 10)
        )

        # Center branch
        self.center_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 3),
            nn.Dropout(dropout)
        )

        # Right branch
        self.right_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(10, 10),
            nn.Dropout(dropout)
        )

        # Merge branch
        # After concatenation, merge & reduce to an 8-dim vector
        self.merge_branch = nn.Sequential(
            nn.Conv2d(64 + 64 + 3, 64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()                    # -> [batch, 5*5*64]
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 64, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run each branch
        x_l = self.left_branch(x)
        x_c = self.center_branch(x)
        x_r = self.right_branch(x)

        # concatenate on channel dim
        x_cat = torch.cat([x_l, x_c, x_r], dim=1)

        # merge path → 8-dim features
        feats = self.merge_branch(x_cat)

        # final logit
        logits = self.classifier(feats)
        return logits.view(-1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 8-D embedding from the classifier."""
        x_l = self.left_branch(x)
        x_c = self.center_branch(x)
        x_r = self.right_branch(x)
        x_cat = torch.cat([x_l, x_c, x_r], dim=1)
        feats = self.merge_branch(x_cat)
        emb = self.classifier[:-1](feats)
        return emb.view(emb.size(0), -1)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))
