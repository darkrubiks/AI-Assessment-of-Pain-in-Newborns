"""
train.py

Author: Leonardo Antunes Ferreira
Date: 10/07/2022

Code for training Deep Learning models.
"""
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
from torch.utils.data import  DataLoader
from datetime import datetime
from tqdm import tqdm

import dataloaders
import models
from validate import validation_metrics
from utils import load_config, write_to_csv


def load_dataset(config):
    # Load the Dataset
    train_dataset = getattr(dataloaders, config['model']+'Dataset') \
                           (img_dir=os.path.join('Datasets','Folds'),
                            fold=config['fold'],
                            mode='Train',
                            soft=config['soft_label'],
                            cache=config['cache'])

    test_dataset = getattr(dataloaders, config['model']+'Dataset') \
                          (img_dir=os.path.join('Datasets','Folds'),
                           fold=config['fold'],
                           mode='Test',
                           cache=config['cache'])
    
    # Batch and Shuffle the Dataset
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['batch_size'], 
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=config['pin_memory'])
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=config['batch_size'], 
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=config['pin_memory'])

    return train_dataloader, test_dataloader

def train(model, dataloader, criterion, optimizer, config):
    # Train for one epoch
    model.train()

    running_loss = 0.0
    preds_list = []
    labels_list = []

    for i,batch in enumerate(dataloader, start=1):
        inputs = batch['image'].to(config['device'])
        labels = batch['label'].to(config['device'])

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
        if config['soft_label']:
            _, labels = torch.max(labels, 1)

        # Statistics
        running_loss += loss.item()
        preds_list += preds.cpu().numpy().tolist()
        labels_list += labels.cpu().numpy().tolist()

        metrics = validation_metrics(labels_list, preds_list)
        metrics.update({'Loss': running_loss/i})

        # Update progress bar
        dataloader.set_postfix(metrics)

    return metrics

def test(model, dataloader, criterion, config):
    # Test for one epoch
    model.eval()

    running_loss = 0.0
    preds_list = []
    labels_list = []

    # Disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for i,batch in enumerate(dataloader, start=1):
            inputs = batch['image'].to(config['device'])
            labels = batch['label'].to(config['device'])

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

    return metrics

def main(config):
    # Define file_name to save epochs results
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M')
    train_log = f"train_log_{config['model']}_{timestamp}.csv"
    test_log = f"test_log_{config['model']}_{timestamp}.csv"

    # Instantiate the model
    model = getattr(models, config['model'])()
    model = model.to(config['device'])

    # Filename to save the model
    model_file_name = f'best_NCNN_fold.pt' # alterar essa porcaria

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), **config['optimizer_hyp'])

    # Define scheduler
    if 'scheduler' in config:
        scheduler = getattr(schedulers, config['scheduler'])(optimizer, **config['scheduler_hyp'])

    counter = 0 # Counter for the number of epochs with no improvement
    best_val_loss = float('inf') # Variable to record the best test loss

    # Load data
    train_dataloader, test_dataloader = load_dataset(config)

    # Start timer
    since = time.time()

    # Training process
    for epoch in range(1,config['epochs']+1):

        # TQDM progress bar
        train_dataloader = tqdm(train_dataloader, unit=' batch', colour='#00ff00', smoothing=0)
        train_dataloader.set_description(f"Train - Epoch [{epoch}/{config['epochs']}]")

        # Train function
        train_metrics = train(model, train_dataloader, criterion, optimizer, config)
        write_to_csv(train_log, **train_metrics)

        # Close TQDM after its iteration
        train_dataloader.close()

        # TQDM progress bar
        test_dataloader = tqdm(test_dataloader, unit=' batch', colour='#00ff00', smoothing=0)
        test_dataloader.set_description(f"Test - Epoch [{epoch}/{config['epochs']}]")
        
        # Test function
        test_metrics = test(model, test_dataloader, criterion, config)
        write_to_csv(test_log, **test_metrics)

        # Close TQDM after its iteration
        test_dataloader.close()

        print()

        # Model saving
        # TODO change to checkpoint?
        epoch_loss = test_metrics['Loss']
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join('models', model_file_name))
        else:
            counter += 1

        # Update scheduler
        if 'scheduler' in config:
            scheduler.step()

        # Check if the stopping criterion is met
        if counter >= config['patience']:
            print(f'Early stopping at epoch {epoch}')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

if __name__=='__main__':

    # Set manual seed
    torch.manual_seed(0)

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='The config .yaml file')
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)