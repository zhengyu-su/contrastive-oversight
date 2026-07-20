'''
Script to create a logger.
'''

import os
import logging
import csv
import json

def setup_logger(exp_name, save_dir):
    """
    Logger setup: logs will be saved to save_dir/train.log and also printed to console.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
        stream_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger

class ExperimentTracker:
    """
    Function to save training/testing metrics to a CSV file for later analysis and plotting.
    """
    def __init__(self, save_dir):
        self.csv_path = os.path.join(save_dir, "metrics.csv")
        self.fieldnames = ['seed','epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_metrics(self, epoch, metrics_dict):
        """
        metrics_dict: {'train_loss': 0.5, 'train_acc': 0.98, ...}
        """
        row = {'epoch': epoch}
        row.update(metrics_dict)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def save_config(self, config_dict):
        """
        Save experiment configuration to a JSON file for reproducibility.
        """
        folder = os.path.dirname(self.csv_path)
        config_path = os.path.join(folder, 'config.json')
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            print(f"✅ Configuration saved to {config_path}")
        except Exception as e:
            print(f"⚠️ Failed to save config: {e}")