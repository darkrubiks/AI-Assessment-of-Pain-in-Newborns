import torch
from torchvision.models import vgg16
import torch.nn as nn

class VGG16ArcFace(nn.Module):
    def __init__(self, num_classes: int = 512, dropout: float = 0.5) -> None:
        super(VGG16ArcFace, self).__init__()
        self.vgg = vgg16(dropout=dropout)
        self.vgg.classifier = nn.Identity()
        self.vgg.avgpool = nn.Identity()
        self.embeddings = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512 * 7 * 7, num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vgg(x)
        logits = self.embeddings(feats)

        return logits