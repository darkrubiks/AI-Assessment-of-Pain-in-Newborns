"""
train.py

Author: Leonardo Antunes Ferreira
Date: 10/07/2022

Code for training  Deep Learning models.
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import  DataLoader
from tqdm import tqdm

import dataloaders
import models
from validate import validation_metrics


def load_dataset(args):
    # Load the Dataset
    train_dataset = getattr(dataloaders, args.model+'Dataset')(img_dir=os.path.join('Datasets','Folds'),
                                                               fold=args.fold,
                                                               mode='Train',
                                                               soft=args.soft,
                                                               cache=args.cache)

    test_dataset = getattr(dataloaders, args.model+'Dataset')(img_dir=os.path.join('Datasets','Folds'),
                                                              fold=args.fold,
                                                              mode='Test',
                                                              cache=args.cache)
    
    # Batch and Shuffle the Dataset
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=args.pin_memory)
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=args.pin_memory)

    return train_dataloader, test_dataloader

def train(model, dataloader, criterion, optimizer, args):
    # Train for one epoch
    model.train()

    running_loss = 0.0
    preds_list = []
    labels_list = []

    for i,batch in enumerate(dataloader, start=1):
        inputs = batch['image'].to(args.device)
        labels = batch['label'].to(args.device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Calculate loss
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Only when using soft labels
        if args.soft:
            _, labels = torch.max(labels, 1)

        # Statistics
        running_loss += loss.item()
        preds_list += preds.cpu().numpy().tolist()
        labels_list += labels.cpu().numpy().tolist()

        metrics = validation_metrics(labels_list, preds_list)
        metrics.update({'Loss': running_loss/i})

        # Update progress bar
        dataloader.set_postfix(metrics)

def test(model, dataloader, criterion, args):
    # Test for one epoch
    model.eval()

    running_loss = 0.0
    preds_list = []
    labels_list = []

    # Disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for i,batch in enumerate(dataloader, start=1):
            inputs = batch['image'].to(args.device)
            labels = batch['label'].to(args.device)

            # Calculate loss
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            preds_list += preds.cpu().numpy().tolist()
            labels_list += labels.cpu().numpy().tolist()

            metrics = validation_metrics(labels_list, preds_list)
            metrics.update({'Loss': running_loss/i})

            # Update progress bar
            dataloader.set_postfix(metrics)

    return running_loss

def main(args):
    # Instantiate the model
    model = getattr(models, args.model)()
    model = model.to(args.device)

    # Filename to save the model
    model_file_name = f'best_NCNN_fold_{args.fold}.pt'

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Define scheduler
    if args.cos_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, verbose=False)

    counter = 0 # Counter for the number of epochs with no improvement
    best_val_loss = float('inf') # Variable to record the best test loss

    # Load data
    train_dataloader, test_dataloader = load_dataset(args)

    # Start timer
    since = time.time()

    # Training process
    for epoch in range(1,args.epochs+1):

        # TQDM progress bar
        train_dataloader = tqdm(train_dataloader, unit=' batch', colour='#00ff00', smoothing=0)
        train_dataloader.set_description(f"Train - Epoch [{epoch}/{args.epochs}]")

        # Train function
        train(model, train_dataloader, criterion, optimizer, args)

        # Close TQDM after its iteration
        train_dataloader.close()

        # TQDM progress bar
        test_dataloader = tqdm(test_dataloader, unit=' batch', colour='#00ff00', smoothing=0)
        test_dataloader.set_description(f"Test - Epoch [{epoch}/{args.epochs}]")
        
        # Test function
        epoch_loss = test(model, test_dataloader, criterion, args)

        # Close TQDM after its iteration
        test_dataloader.close()

        # Model saving
        # TODO change to checkpoint?
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join('models', model_file_name))
        else:
            counter += 1

        # Update scheduler
        if args.cos_lr:
            scheduler.step()

        # Check if the stopping criterion is met
        if counter >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

if __name__=='__main__':

    # Set manual seed
    torch.manual_seed(0)

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['NCNN', 'VGGNB'], help='The model to be trained')
    parser.add_argument('--fold', type=str, default='0', help='Fold number') # TODO trocar para diretorio
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--label_smoothing', type=float, default=0, help='Label smoothing epsilon')
    parser.add_argument('--cos_lr', action='store_true', help='Cosine annealing scheduler')
    parser.add_argument('--soft', action='store_true', help='Use Soft Labels generated from NFCS. Only applies to UNIFESP dataset. Dont use with Label Smoothing!')
    parser.add_argument('--cache', action='store_true', help='Cache images on RAM')
    parser.add_argument('--device', type=str, default='cpu', help='Which device to use "cpu" or "cuda"')
    parser.add_argument('--pin_memory', action='store_true', help='Dataloader pinned memory function')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker to use in Dataloader')
    args = parser.parse_args()

    main(args)