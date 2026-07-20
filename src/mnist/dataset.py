'''
Script to load the MNIST dataset and prepare it for training and testing.
'''

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

class CounterfactualDataset(Dataset):
    """
    Load human counterfactual data
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. Dimension should be (1, 28, 28) and type should be torch.Tensor
        if not isinstance(sample, torch.Tensor):
            sample = torch.from_numpy(sample).float()
        
        # (28, 28, 1) -> (1, 28, 28)
        if sample.ndimension() == 3 and sample.shape[-1] == 1:
            sample = sample.permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)
        
        # 2. Transformation
        if self.transform:
            sample = self.transform(sample)
            
        # 3. Get int label
        label = self.labels[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        return sample, int(label)

def get_mnist_transforms(transform_type):
    """
    Transforms for the counterfactual dataset, with optional augmentation.
    """
    if transform_type == "cf_augment":
        return transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    elif transform_type == "train_augment":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[1.0]),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    elif transform_type == "test":
        # transform original data to tensor and between [-0.5,0.5]
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[1.0])
        ])

def load_combined_mnist_dataset(cf_data, cf_labels, batch_size=256):
    """
    Combine original training data with human cf data
    """
    # Load original mnist training data
    standard_trainset = datasets.MNIST(root='../data', train=True, download=True,
                                       transform=get_mnist_transforms("train_augment"))

    # Load cf dataset
    cf_transform = get_mnist_transforms("cf_augment")
    cf_dataset = CounterfactualDataset(cf_data, cf_labels, transform=cf_transform)

    combined_dataset = ConcatDataset([standard_trainset, cf_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    standard_loader = DataLoader(standard_trainset, batch_size=batch_size, shuffle=True)
    
    return combined_loader, standard_loader

def load_misclassified_test_dataset(user_indices, batch_size=1):
    """
    Load original MNIST test samples that are misclassified
    """
    testset = load_full_testset()
    misclassified_subset = Subset(testset, user_indices)
    misclassified_loader = DataLoader(misclassified_subset, batch_size=batch_size, shuffle=False)
    
    return misclassified_loader

def load_full_testset():
    testset = datasets.MNIST(root='../data', train=False, download=True,
                             transform=get_mnist_transforms("test"))
    return testset

def load_user_indices():
    '''
    load the indices in MNIST test data for the selected samples to generate human cfs
    '''
    cf_indices = np.load('./data/mnist_cf/user_indices_mnist.npy')
    return cf_indices

def user_indices_to_label():
    cf_indices = load_user_indices()
    full_testset = load_full_testset()
    cf_labels = full_testset.targets[cf_indices]
    return cf_labels