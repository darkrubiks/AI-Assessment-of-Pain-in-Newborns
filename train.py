"""
train.py

Author: Leonardo Antunes Ferreira
Date: 10/07/2022

Code for training Deep Learning models.
"""
import argparse
import os
import shutil
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloaders
import models
from utils.utils import load_config, write_to_csv, create_folder
from validate import validation_metrics, validation_plots, calibration_metrics

# Get current directory
ROOT = os.getcwd()
SAVE_DIR = os.path.join(ROOT, 'experiments')
    

def label_smooth_binary_cross_entropy(outputs, labels, epsilon=0.0):
    # Custom binary cross-entropy loss with label smoothing.
    epsilon = 1.0 if epsilon > 1.0 else epsilon
    smoothed_labels = (1 - epsilon) * labels + epsilon / 2
    loss = nn.BCEWithLogitsLoss()(outputs, smoothed_labels)

    return loss


def load_dataset(config):
    # Load the Dataset
    train_dataset = getattr(dataloaders, config['model']+'Dataset') \
                           (path=config['path_train'],
                            soft=config['soft_label'],
                            cache=config['cache'])

    test_dataset = getattr(dataloaders, config['model']+'Dataset') \
                          (path=config['path_test'],
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


def train(model, dataloader, optimizer, config):
    # Train for one epoch
    model.train()

    running_loss = 0.0
    preds_list = torch.empty(0, device=config['device'])
    probs_list = torch.empty(0, device=config['device'])
    labels_list = torch.empty(0, device=config['device'])
 
    for i,batch in enumerate(dataloader, start=1):
        inputs = batch['image'].to(config['device'])
        labels = batch['label'].to(config['device'])

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Calculate loss
        outputs = model(inputs)
        loss = label_smooth_binary_cross_entropy(outputs, labels, epsilon=config['label_smoothing'])

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Only when using soft labels
        if config['soft_label']:
            labels = torch.ge(labels, 0.5).type(torch.int)

        # Statistics
        probs = F.sigmoid(outputs).detach()
        preds = torch.ge(probs, 0.5).type(torch.int)
        running_loss += loss.item()
        preds_list = torch.cat([preds_list, preds])
        probs_list = torch.cat([probs_list, probs])
        labels_list = torch.cat([labels_list, labels])

        # Only update after 20 batches
        if i % 20 == 0:
            metrics = validation_metrics(preds_list.cpu().numpy(), probs_list.cpu().numpy(), labels_list.cpu().numpy())
            metrics.update(calibration_metrics(probs_list.cpu().numpy(), labels_list.cpu().numpy()))
            metrics.update({'Loss': running_loss/i})
            # Update progress bar
            dataloader.set_postfix(metrics)

    return metrics


def test(model, dataloader, config):
    # Test for one epoch
    model.eval()

    running_loss = 0.0
    preds_list = torch.empty(0, device=config['device'])
    probs_list = torch.empty(0, device=config['device'])
    labels_list = torch.empty(0, device=config['device'])

    # Disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for i,batch in enumerate(dataloader, start=1):
            inputs = batch['image'].to(config['device'])
            labels = batch['label'].to(config['device'])

            # Calculate loss
            outputs = model(inputs)
            loss = label_smooth_binary_cross_entropy(outputs, labels, epsilon=config['label_smoothing'])

            # Statistics
            probs = F.sigmoid(outputs).detach()
            preds = torch.ge(probs, 0.5).type(torch.int)
            running_loss += loss.item()
            preds_list = torch.cat([preds_list, preds])
            probs_list = torch.cat([probs_list, probs])
            labels_list = torch.cat([labels_list, labels])

            metrics = validation_metrics(preds_list.cpu().numpy(), probs_list.cpu().numpy(), labels_list.cpu().numpy())
            metrics.update(calibration_metrics(probs_list.cpu().numpy(), labels_list.cpu().numpy()))
            metrics.update({'Loss': running_loss/i})

            # Update progress bar
            dataloader.set_postfix(metrics)

    return metrics


def main(config):
    # Define experiment name to save results
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M')
    experiment_dir = os.path.join(SAVE_DIR, f"{timestamp}_{config['model']}")
    create_folder(experiment_dir)
    for folder in ['Logs', 'Model', 'Results']:
        create_folder(os.path.join(experiment_dir, folder))
   
    # Log names
    train_log = os.path.join(experiment_dir, 'Logs', f"train_log.csv")
    test_log = os.path.join(experiment_dir, 'Logs', f"test_log.csv")

    # Filename to save the model
    model_file = os.path.join(experiment_dir, 'Model', f"best_model.pt")
    shutil.copy(args.config, os.path.join(experiment_dir, 'Model', 'config.yaml')) # Copy .yaml file

    # Instantiate the model
    model = getattr(models, config['model'])()
    model = model.to(config['device'])

    # Define optimizer
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
        train_metrics = train(model, train_dataloader, optimizer, config)
        write_to_csv(train_log, **train_metrics)

        # Close TQDM after its iteration
        train_dataloader.close()

        # TQDM progress bar
        test_dataloader = tqdm(test_dataloader, unit=' batch', colour='#00ff00', smoothing=0)
        test_dataloader.set_description(f"Test  - Epoch [{epoch}/{config['epochs']}]")
        
        # Test function
        test_metrics = test(model, test_dataloader, config)
        write_to_csv(test_log, **test_metrics)

        # Close TQDM after its iteration
        test_dataloader.close()

        print()

        # Model saving
        epoch_loss = test_metrics['Loss']
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            counter = 0
            torch.save(model.state_dict(), model_file)
        else:
            counter += 1

        # Update scheduler
        if 'scheduler' in config:
            scheduler.step()

        # Check if the stopping criterion is met
        if counter >= config['patience']:
            print(f'Early stopping at epoch {epoch}\n')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    
    # Run validation plots on the best model
    print(f"Saving Results to {os.path.join(experiment_dir, 'Results')}\n")
    model.eval()
    model.load_state_dict(torch.load(model_file))

    preds_list = torch.empty(0, device=config['device'])
    labels_list = torch.empty(0, device=config['device'])
    probs_list = torch.empty(0, device=config['device'])

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['image'].to(config['device'])
            labels = batch['label'].to(config['device'])

            # Calculate loss
            outputs = model(inputs)
            # Statistics
            probs = F.sigmoid(outputs)
            preds = torch.ge(probs, 0.5).type(torch.int)
            
            preds_list = torch.cat([preds_list, preds])
            labels_list = torch.cat([labels_list, labels])
            probs_list = torch.cat([probs_list, probs])

    validation_plots(preds_list.cpu().numpy(), 
                     probs_list.cpu().numpy(), 
                     labels_list.cpu().numpy(), 
                     path=os.path.join(experiment_dir, 'Results'))

if __name__=='__main__':

    # Set manual seed
    torch.manual_seed(1234)

    create_folder(SAVE_DIR)

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='The config .yaml file')
    args = parser.parse_args()

    config = load_config(args.config)

    if config['soft_label'] and config['label_smoothing'] !=0:
        print('Please dont use soft labels and label smoothing together!')
        print('Aborting...')
        exit(0)

    if config['cache'] and config['num_workers'] > 0:
        print('Number of workers should be zero to use cache!')
        print('Aborting...')
        exit(0)

    main(config)