"""
train.py

Author: Leonardo Antunes Ferreira
Date:13/02/2022

Code for training a Deep Learning model to classify pain and no-pain face images
of newborns
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.utils.data import  DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset_maker import VGGNBDataset
from models.VGGNB import VGGNB


#  Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=str, default='0', help='Fold number')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
args = parser.parse_args()

# Load the Dataset
train_dataset = VGGNBDataset(os.path.join('Datasets','Folds'), args.fold, 'Train')
test_dataset = VGGNBDataset(os.path.join('Datasets','Folds'), args.fold, 'Test')
# Batch and Shuffle the Dataset
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

# Instantiate the VGGNB model
model = VGGNB()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metrics
criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)

model = model.to(device)

since = time.time()

num_epochs = args.epochs # Total training epochs

patience = args.patience # Number of epochs with no improvement
counter = 0 # Counter for the number of epochs with no improvement

best_val_loss = float('inf')
best_val_acc = float('inf')
best_val_f1 = float('inf')
best_val_precision = float('inf')
best_val_recall = float('inf')

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
                best_val_f1 = f1_score(labels.cpu(), preds.cpu())
                best_val_precision = precision_score(labels.cpu(), preds.cpu())
                best_val_recall = recall_score(labels.cpu(), preds.cpu())
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
print(f'F1 Score: {best_val_f1:.4f}')
print(f'Precision Score: {best_val_precision:.4f}')
print(f'Recall Score: {best_val_recall:.4f}')