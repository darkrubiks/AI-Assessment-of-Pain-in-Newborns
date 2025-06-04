import torch
import torch.nn as nn
from models.VGG16ArcFace import VGG16ArcFace
from models.NCNNArcFace import NCNNArcFace


class PainClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = VGG16ArcFace()
        self.num_classes = num_classes
        self.dropout = dropout

        checkpoint = torch.load('D:\\Doutorado\\Cassia Test\\ms1mv3_arcface_VGG16\\model.pt', weights_only=False)
        self.backbone.load_state_dict(checkpoint)

        # Freeze the arcface model parameters
        for param in self.backbone.parameters():
            param.requires_grad  = False

        # Fine tune last conv block
        #for param in self.backbone.vgg.features[24:].parameters():
            #param.requires_grad = True

        #for param in self.backbone.merge_branch.parameters():
            #param.requires_grad  = True 

        # Fine tune embeddings
        #for param in self.backbone.classifier.parameters():
            #param.requires_grad = True

        # Zero-shot with features
        #self.backbone.classifier = nn.Sequential(
        #    nn.Dropout(self.dropout),
            #nn.Linear(7 * 7 * 512, 1)
        #)

        # Zero-shot with embeddings
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(512, 1)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using the arcface model
        x = self.backbone(x)

        # Classify
        #x = self.classifier(x)

        return x.view(-1)
