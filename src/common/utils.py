import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def load_cf_data(g1_path, g2_path, indices_path, num_samples=50):
    """
    Load  *num_samples* random selected cf data from .npy files
    g1_path: .npy path for cf data (group 1)
    g2_path: .npy path for cf data (group 2)
    indices_path: .npy path for 50 misclassified image indices in original mnist training set
    """

    user_explanations_g1 = np.load(g1_path)
    user_explanations_g2 = np.load(g2_path)
    user_indices = np.load(indices_path)
    
        
    return user_explanations_g1, user_explanations_g2, user_indices

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
Visualization for original data
'''

def calculate_accuracy_matrix(predicted_labels, true_labels, num_classes=10):
    num_users, num_samples = predicted_labels.shape
    accuracy_matrix = np.zeros((num_users, num_classes))
    class_sample_counts = np.zeros(num_classes)

    for user_idx in range(num_users):
        user_predictions = predicted_labels[user_idx]
        
        for class_id in range(num_classes):
            
            indices_of_class = (true_labels == class_id)
            
            if np.sum(indices_of_class) == 0:
                accuracy_matrix[user_idx, class_id] = 0 
                continue

            predictions_for_class = user_predictions[indices_of_class]
            true_labels_for_class = true_labels[indices_of_class]
            
            correct_count = np.sum(predictions_for_class == true_labels_for_class)
            total_count = len(true_labels_for_class)
            
            accuracy_matrix[user_idx, class_id] = correct_count / total_count

            class_sample_counts[class_id] += total_count

    total_correct = np.sum(predicted_labels == true_labels)
    total_samples = predicted_labels.size 
    total_average_acc = total_correct / total_samples
            
    return accuracy_matrix, class_sample_counts, total_average_acc

# Calculate group wise accuracy
def calculate_group_wise_accuracy(accuracy_matrix):
    group_wise_accuracy = np.mean(accuracy_matrix, axis=0)
    return group_wise_accuracy