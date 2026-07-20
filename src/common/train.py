import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import datetime

import src.mnist.models as models
from src.common.utils import get_device, set_seed
from src.common.logger import setup_logger, ExperimentTracker

def train_one_epoch(model, loader, optimizer, criterion, device, logger, adv_train=False, eps=0.15):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if adv_train:
            # --- Generate adv examples ---
            # requires_grad for calculating the gradient
            inputs.requires_grad = True
            
            # Calculate original loss
            outputs_orig = model(inputs)
            loss_orig = criterion(outputs_orig, labels)
            
            # Calculate d(loss)/d(inputs)
            model.zero_grad()
            loss_orig.backward()
            
            # FGSM：x_adv = x + eps * sign(grad)
            # .detach() for not letting it impact the training itself
            adv_inputs = inputs + eps * inputs.grad.sign()
            adv_inputs = torch.clamp(adv_inputs, 0, 1).detach()
            
            # --- Add adv to training data ---
            combined_inputs = torch.cat([inputs, adv_inputs], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            # Forward pass again
            optimizer.zero_grad()
            combined_outputs = model(combined_inputs)
            loss = criterion(combined_outputs, combined_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = combined_outputs.max(1)
            total += combined_labels.size(0)
            correct += predicted.eq(combined_labels).sum().item()
        
        else:
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        
    avg_loss = running_loss / len(loader)
    acc = correct / total if total > 0 else 0
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    acc = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total if total > 0 else 0
    val_loss /= len(loader)
            
    return correct, total, acc, val_loss

def run_training(model, train_loader, val_loader, epochs, lr, device, logger, tracker, seed, adv_train, eps):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    logger.info(f"Training started on device: {device}")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, logger
                                                , adv_train, eps)
        
        # Val/Test
        val_correct, val_total, val_acc, val_loss = validate(model, val_loader, criterion, device)
        
        # Logging
        logger.info(f"Epoch {epoch}/{epochs}: "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}% | "
                    f"Val/Test Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%")
        
        # Save metrics to CSV for later analysis
        tracker.log_metrics(epoch, {
            'seed': seed,
            'train_loss': train_loss, 
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        

    return train_acc, val_acc, model

def resample_model(model_class):
    return model_class()

def run_random_seeds(exp_name, run_folder, train_loader, tracker, device, 
                     lr,
                     model_class,
                     epochs=3,
                     seeds = [42, 123, 7],
                     adv_train=False,
                     eps=0.15,
                     with_cf=False):
    all_results = []

    os.makedirs(run_folder, exist_ok=True)
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    logger = setup_logger(exp_name, run_folder)
    logger.info(f"Running training with seeds: {seeds}")

    for run_idx, seed in enumerate(seeds):
        set_seed(seed)

        logger.info(f"--- Starting Run {run_idx + 1} with seed {seed} ---")
        logger.info(f"CF: {with_cf}, adv: {adv_train}")

        model = model_class().to(device)
        train_acc, val_acc, model = run_training(model, train_loader, train_loader, epochs, lr=lr, device=device, logger=logger, tracker=tracker, seed=seed,
                                                 adv_train=adv_train, eps=eps)

        PATH = f"{run_folder}/model_seed_{seed}.pth"
        torch.save(model.state_dict(), PATH)

        all_results.append(val_acc)
    
    all_results = np.array(all_results)
    mean_acc = np.mean(all_results)
    std_acc = np.std(all_results)

    logger.info(f"Final Result (on test set): {100 * mean_acc:.2f}% ± {100 * std_acc:.2f}%")

