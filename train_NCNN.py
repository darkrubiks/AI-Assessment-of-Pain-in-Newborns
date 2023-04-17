"""
NCNN_train.py

Author: Leonardo Antunes Ferreira
Date: 13/02/2022

Code for training the NCNN model to classify pain and no-pain face images of 
newborns.
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import  DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataloaders import NCNNDataset
from models import NCNN


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=str, default='0', help='Fold number')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
parser.add_argument('--label-smoothing', type=float, default=0, help='Label smoothing epsilon')
parser.add_argument('--cos-lr', action='store_true', help='Cosine annealing scheduler')
parser.add_argument('--cache', action='store_true', help='Cache images on RAM')
args = parser.parse_args()

# Set manual seed
torch.manual_seed(0)
# Search for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Dataset
train_dataset = NCNNDataset(img_dir=os.path.join('Datasets','Folds'),
                            fold=args.fold,
                            mode='Train',
                            cache=args.cache)

test_dataset = NCNNDataset(img_dir=os.path.join('Datasets','Folds'),
                           fold=args.fold,
                           mode='Test',
                           cache=args.cache)
# Batch and Shuffle the Dataset
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Instantiate the VGGNB model
model = NCNN()
model = model.to(device)

model_file_name = f'best_NCNN_fold_{args.fold}.pt'

# Metrics
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)
if args.cos_lr:
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, verbose=False)

num_epochs = args.epochs # Total training epochs

patience = args.patience # Number of epochs with no improvement
counter = 0 # Counter for the number of epochs with no improvement

best_val_loss = float('inf')
best_val_acc = float('inf')

dataloader = {'train':train_dataloader, 'test':test_dataloader}
dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}

since = time.time()

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
                torch.save(model.state_dict(), os.path.join('models', model_file_name))
            else:
                counter += 1

    print()
    if args.cos_lr:
        scheduler.step()

    # Check if the stopping criterion is met
    if counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

# Run validation metrics for the best model
print('-' * 10)
print('Running validation metrics on the best model...')

model.load_state_dict(torch.load(os.path.join('models', model_file_name)))
model.eval()

labels = []
preds = []

for test_sample in test_dataset:
    image = test_sample['image'].to(device)
    label = test_sample['label'] # Already returns an int
    # The torch.max function also returns the maximum value, but for now it is _
    _, pred = torch.max(model.predict(image.unsqueeze(dim=0)), 1)

    labels.append(label)
    preds.append(pred.cpu().numpy())

best_val_acc = accuracy_score(labels, preds)
best_val_f1 = f1_score(labels, preds)
best_val_precision = precision_score(labels, preds)
best_val_recall = recall_score(labels, preds)

print()
print(f'Best test Acc: {best_val_acc:4f}')
print(f'F1 Score: {best_val_f1:.4f}')
print(f'Precision Score: {best_val_precision:.4f}')
print(f'Recall Score: {best_val_recall:.4f}')