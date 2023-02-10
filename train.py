import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.data import  DataLoader
from dataset_maker import NBDatasetVGGFace

train_dataset = NBDatasetVGGFace('Datasets\\Folds', '0', 'Train')
test_dataset = NBDatasetVGGFace('Datasets\\Folds', '0', 'Test')

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

class VGGNB(nn.Module):
    def __init__(self):
        super(VGGNB, self).__init__()

        self.VGGFace = torch.load('models\\VGG_face_original_model.pt')

        for param in self.VGGFace.parameters():
            param.requires_grad  = False

        self.VGGFaceFeatures = self.VGGFace.features

        self.Flatten = nn.Flatten()

        self.Dropout = nn.Dropout(0.5)

        self.FC1 = nn.Linear(512 * 7 * 7, 512)
        self.FC2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 2)


    def forward(self, x):
        x = self.VGGFaceFeatures(x)
        x = self.Flatten(x)
        x = F.relu(self.FC1(x))
        x = self.Dropout(x)
        x = F.relu(self.FC2(x))
        x = self.Dropout(x)
        x = self.output(x)

        return x
    
    def predict(self, x):
        return F.softmax(self.forward(x))
    
model = VGGNB()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)

model.to(device)

for epoch in range(25):
    for i, batch in enumerate(train_dataloader, 0):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

    print(f"Epoch: {epoch+1}         Loss: {loss:.4f}")

