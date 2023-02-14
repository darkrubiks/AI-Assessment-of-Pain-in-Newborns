import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import  DataLoader
from dataset_maker import VGGNBDataset
from models.models import VGGNB

train_dataset = VGGNBDataset('Datasets\\Folds', '0', 'Train')
test_dataset = VGGNBDataset('Datasets\\Folds', '0', 'Test')

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model = VGGNB()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-5)

model = model.to(device)

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

