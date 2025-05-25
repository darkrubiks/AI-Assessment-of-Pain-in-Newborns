import torch
import torch.nn as nn
from models.VGG16 import VGG16


class PainClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.arcface_model = VGG16()
        self.num_classes = num_classes
        self.dropout = dropout

        

        checkpoint = torch.load('D:\\Doutorado\\Cassia Test\\ms1mv3_arcface_VGG16\\model.pt', weights_only=False)
        self.arcface_model.load_state_dict(checkpoint)

        for param in self.arcface_model.parameters():
            param.requires_grad  = False

        for param in self.arcface_model.vgg.features[24:].parameters():
            param.requires_grad = True

        #for param in self.arcface_model.classifier.parameters():
            #param.requires_grad = True


        # Classifier head
        self.arcface_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, num_classes),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using the arcface model
        x = self.arcface_model(x)

        # Classify
        #x = self.classifier(x)

        return x.view(-1)
