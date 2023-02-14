"""
train.py

Author: Leonardo Antunes Ferreira
Date:13/02/2022

Code for training a Deep Learning model to classify pain and no-pain face images
of newborns
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import  DataLoader
from dataset_maker import VGGNBDataset
from models.models import VGGNB


# Load the Dataset
train_dataset = VGGNBDataset(os.path.join('Datasets','Folds'), '0', 'Train')
test_dataset = VGGNBDataset(os.path.join('Datasets','Folds'), '0', 'Test')
# Batch and Shuffle the Dataset
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Instantiate the VGGNB model
model = VGGNB()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metrics
criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=1e-5, weight_decay=1e-5)

model = model.to(device)

since = time.time()

num_epochs = 25 # Total training epochs

# Define the stopping criterion
patience = 5 # Number of epochs with no improvement
counter = 0 # Counter for the number of epochs with no improvement
best_val_loss = float('inf')
best_val_acc = float('inf')

dataloader = {'train':train_dataloader, 'test':test_dataloader}
dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}

# Start Training and Testing
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Each epoch has a training and testing phase
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for batch in dataloader[phase]:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            # Track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Save best test model
        if phase == 'test':
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_val_acc = epoch_acc
                counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                counter += 1

    print()

    # Check if the stopping criterion is met
    if counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best test Acc: {best_val_acc:4f}')