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
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
from torch.utils.data import DataLoader

from dataloaders import BaseDataset
import models
from utils.utils import load_config, write_to_csv, create_folder
from validate import validation_metrics, validation_plots, calibration_metrics

# Configure native logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get current directory and save directory
ROOT = os.getcwd()
SAVE_DIR = os.path.join(ROOT, 'experiments')


def format_metrics(metrics, precision=4):
    return {k: f"{v:.{precision}f}" if isinstance(v, (float, int)) else v
            for k, v in metrics.items()}


def label_smooth_binary_cross_entropy(outputs, labels, epsilon=0.0):
    # Custom binary cross-entropy loss with label smoothing.
    epsilon = 1.0 if epsilon > 1.0 else epsilon
    smoothed_labels = (1 - epsilon) * labels + epsilon / 2
    loss = nn.BCEWithLogitsLoss()(outputs, smoothed_labels)
    return loss


def load_dataset(config):
    # Load the Dataset
    train_dataset = BaseDataset(model_name=config['model'],
                                img_dir=config['path_train'],
                                soft=config['soft_label'],
                                cache=config['cache']
    )

    test_dataset = BaseDataset(model_name=config['model'],
                                img_dir=config['path_test'],
                                cache=config['cache']
    )
    
    # Batch and Shuffle the Dataset
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    return train_dataloader, test_dataloader


def train(model, dataloader, optimizer, config):
    # Train for one epoch
    model.train()

    running_loss = 0.0
    preds_list = torch.empty(0, device=config['device'])
    probs_list = torch.empty(0, device=config['device'])
    labels_list = torch.empty(0, device=config['device'])

    # Initialize counters for image processing speed
    total_images = 0
    epoch_start_time = time.time()
    
    for i, batch in enumerate(dataloader, start=1):
        inputs = batch['image'].to(config['device'])
        labels = batch['label'].to(config['device'])

        # Count images processed in this batch
        batch_size = inputs.size(0)
        total_images += batch_size

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Calculate loss
        outputs = model(inputs)
        loss = label_smooth_binary_cross_entropy(outputs, labels, epsilon=config['label_smoothing'])

        # Backpropagation
        loss.backward()
        optimizer.step()

        # When using soft labels, convert to hard labels
        if config['soft_label']:
            labels = torch.ge(labels, 0.5).type(torch.int)

        # Statistics
        probs = torch.sigmoid(outputs).detach()  # Using torch.sigmoid over F.sigmoid (deprecated)
        preds = torch.ge(probs, 0.5).type(torch.int)
        running_loss += loss.item()
        preds_list = torch.cat([preds_list, preds])
        probs_list = torch.cat([probs_list, probs])
        labels_list = torch.cat([labels_list, labels])

    # Compute epoch elapsed time and images per second
    epoch_time = time.time() - epoch_start_time
    images_per_sec = total_images / epoch_time if epoch_time > 0 else 0

    # Compute final metrics after processing all batches
    final_metrics = validation_metrics(
        preds_list.cpu().numpy(),
        probs_list.cpu().numpy(),
        labels_list.cpu().numpy()
    )
    final_metrics.update(calibration_metrics(
        probs_list.cpu().numpy(),
        labels_list.cpu().numpy()
    ))
    final_metrics.update({'Loss': running_loss / i,
                          'Images/s': images_per_sec})

    return final_metrics


def test(model, dataloader, config):
    # Test for one epoch
    model.eval()

    running_loss = 0.0
    preds_list = torch.empty(0, device=config['device'])
    probs_list = torch.empty(0, device=config['device'])
    labels_list = torch.empty(0, device=config['device'])

    # Disable gradient computation
    with torch.no_grad():
        for i, batch in enumerate(dataloader, start=1):
            inputs = batch['image'].to(config['device'])
            labels = batch['label'].to(config['device'])

            # Calculate loss
            outputs = model(inputs)
            loss = label_smooth_binary_cross_entropy(outputs, labels, epsilon=config['label_smoothing'])

            # Statistics
            probs = torch.sigmoid(outputs)
            preds = torch.ge(probs, 0.5).type(torch.int)
            running_loss += loss.item()
            preds_list = torch.cat([preds_list, preds])
            probs_list = torch.cat([probs_list, probs])
            labels_list = torch.cat([labels_list, labels])

    # Compute final metrics after processing all batches
    final_metrics = validation_metrics(
        preds_list.cpu().numpy(),
        probs_list.cpu().numpy(),
        labels_list.cpu().numpy()
    )
    final_metrics.update(calibration_metrics(
        probs_list.cpu().numpy(),
        labels_list.cpu().numpy()
    ))
    final_metrics.update({'Loss': running_loss / i})
   
    return final_metrics


def main(config):
    # Define experiment name to save results
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M')
    experiment_dir = os.path.join(SAVE_DIR, f"{timestamp}_{config['model']}")
    create_folder(experiment_dir)
    for folder in ['Logs', 'Model', 'Results']:
        create_folder(os.path.join(experiment_dir, folder))
   
    # Log file names
    train_log = os.path.join(experiment_dir, 'Logs', "train_log.csv")
    test_log = os.path.join(experiment_dir, 'Logs', "test_log.csv")

    # Filename to save the model
    model_file = os.path.join(experiment_dir, 'Model', "best_model.pt")
    shutil.copy(args.config, os.path.join(experiment_dir, 'Model', 'config.yaml'))  # Copy config file

    # Instantiate the model and send to device
    model = getattr(models, config['model'])()

    # TODO pre trained NCNN
    checkpoint = torch.load('D:/Doutorado/Cassia Test/checkpoints_NCNN/checkpoint_99.pth', weights_only=False)
    model.load_state_dict(checkpoint['model'])

    """
    model.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 64, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 1)
        )
    """

    model = model.to(config['device'])

    # Define optimizer and scheduler (if any)
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), **config['optimizer_hyp'])
    if 'scheduler' in config:
        scheduler = getattr(schedulers, config['scheduler'])(optimizer, **config['scheduler_hyp'])

    counter = 0  # For early stopping
    best_val_loss = float('inf')  # To record the best test loss

    # Load data
    train_dataloader, test_dataloader = load_dataset(config)

    # Start timer
    since = time.time()

    # Training process
    for epoch in range(1, config['epochs'] + 1):
        logger.info(f"Starting Epoch [{epoch}/{config['epochs']}]")

        # Training phase
        train_metrics = train(model, train_dataloader, optimizer, config)
        write_to_csv(train_log, **train_metrics)
        logger.info(f"Finished Training Epoch {epoch}: {format_metrics(train_metrics)}")

        # Testing phase
        test_metrics = test(model, test_dataloader, config)
        write_to_csv(test_log, **test_metrics)
        logger.info(f"Finished Testing Epoch {epoch}: {format_metrics(test_metrics)}")

        # Early stopping and model saving
        epoch_loss = test_metrics['Loss']
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            counter = 0
            torch.save(model.state_dict(), model_file)
            logger.info(f"New best model saved with loss {best_val_loss:.4f}")
        else:
            counter += 1
            logger.info(f"No improvement for {counter} epoch(s).")

        if 'scheduler' in config:
            scheduler.step()

        if counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    time_elapsed = time.time() - since
    logger.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Run validation plots on the best model
    logger.info(f"Saving Results to {os.path.join(experiment_dir, 'Results')}")
    model.eval()
    model.load_state_dict(torch.load(model_file))

    preds_list = torch.empty(0, device=config['device'])
    labels_list = torch.empty(0, device=config['device'])
    probs_list = torch.empty(0, device=config['device'])

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['image'].to(config['device'])
            labels = batch['label'].to(config['device'])

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = torch.ge(probs, 0.5).type(torch.int)
            
            preds_list = torch.cat([preds_list, preds])
            labels_list = torch.cat([labels_list, labels])
            probs_list = torch.cat([probs_list, probs])

    validation_plots(
        preds_list.cpu().numpy(), 
        probs_list.cpu().numpy(), 
        labels_list.cpu().numpy(), 
        path=os.path.join(experiment_dir, 'Results')
    )


if __name__ == '__main__':
    # Set manual seed
    torch.manual_seed(1234)

    create_folder(SAVE_DIR)

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='The config .yaml file')
    args = parser.parse_args()

    config = load_config(args.config)
        
    if config['soft_label'] != "None" and config['label_smoothing'] !=0:
        logger.error('Please dont use soft labels and label smoothing together!')
        exit(0)


    main(config)
