'''
Script to load QuickDraw dataset
'''

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('data/quickdraw/quickdraw_x_train.npy')
y_train = np.load('data/quickdraw/quickdraw_y_train.npy')
x_test = np.load('data/quickdraw/quickdraw_x_test.npy')
y_test = np.load('data/quickdraw/quickdraw_y_test.npy')

misclassification_indices_test_set = [2847, 774, 5494, 13769, 14064, 11587, 11900, 12119, 12853, 13119, 622, 733,
                                      6131, 1843, 2174, 2605, 2823, 3561, 3651, 5469, 6381, 6549, 9132, 9493,
 10135, 10443, 13594, 11498, 13575,  7624]        # Calculated by comparing min edit samples with true labels

QD_CLASS_NAMES = {
    0: "giraffe",
    1: "mushroom",
    2: "bicycle",
    3: "helicopter",
    4: "pizza"
}

misclassified_imgs = np.load('data/quickdraw_cf/quickdraw_misclassified_image_user_set.npy')
predicted_labels = np.load('data/quickdraw_cf/quickdraw_predicted_labels.npy')
true_labels = np.load('data/quickdraw_cf/quickdraw_true_labels.npy')
cf_imgs = np.load('data/quickdraw_cf/user_explanations_quickdraw.npy')

train_transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[1.0]), 
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

test_transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[1.0]),
])


# x_train shape: (35000, 1, 28, 28) (N, H, W, C)
# y_train shape: (35000, 5) -> need to convert shape to (35000,)

x_train_tensor = torch.tensor(x_train).float()   # (35000, 1, 28, 28)
x_train_t = x_train_tensor.permute(0, 3, 1, 2)  # (35000, 28, 28, 1) -> (35000, 1, 28, 28)
y_train_labels = np.argmax(y_train, axis=1)  # (35000, 5) -> (35000,)
y_train_tensor = torch.tensor(y_train_labels).long()

x_test_tensor = torch.tensor(x_test).float()    
x_test_t = x_test_tensor.permute(0, 3, 1, 2)   
y_test_labels = np.argmax(y_test, axis=1)      
y_test_tensor = torch.tensor(y_test_labels).long()

class QuickDrawDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
class QuickDrawCFDataset(Dataset):
    def __init__(self, cf_imgs, true_labels, transform=None):        
        self.data = torch.tensor(cf_imgs).float().permute(1, 0, 4, 2, 3).reshape(-1, 1, 28, 28)
        num_total_imgs = self.data.shape[0] # 30 samples * 5 users = 150 images
        
        labels_tensor = torch.tensor(true_labels).long().flatten()
        num_input_labels = labels_tensor.shape[0]

        # --- Only repeat the labels when needed ---
        if num_input_labels == num_total_imgs:
            # If already 150
            self.targets = labels_tensor
        elif num_input_labels == (num_total_imgs // 5):
            # If only 30, repeat each label 5 times to match 150
            self.targets = labels_tensor.repeat_interleave(5)
        else:
            print(f"⚠️ Warning: inccorect number of labels:  {num_input_labels}. Adjusting to match {num_total_imgs} images.")
            if num_input_labels > num_total_imgs:
                self.targets = labels_tensor[:num_total_imgs]
            else:
                # If too few labels, fill the rest with zeros (or any default label)
                self.targets = torch.zeros(num_total_imgs).long()
                self.targets[:num_input_labels] = labels_tensor
            
        if len(self.data) != len(self.targets):
            raise ValueError(f"Dimensions do not match! {len(self.data)} images, {len(self.targets)} labels.")
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
def get_cf_splits(cf_dataset, user_indices):
    """
    cf_dataset: Full CF dataset with shape (150, 1, 28, 28) and corresponding labels (150,)
    user_indices: pre-generated array of shape (30,) that indicates which user is selected for training for each of the 30 samples
    """
    train_indices = []
    test_indices = []
    
    for s_idx in range(30):
        # Calculate the base index for current sample
        base_pos = s_idx * 5
        train_user_offset = user_indices[s_idx]
        
        for u_idx in range(5):
            global_idx = base_pos + u_idx
            if u_idx == train_user_offset:
                train_indices.append(global_idx)
            else:
                test_indices.append(global_idx)
                
    train_subset = Subset(cf_dataset, train_indices)
    test_subset = Subset(cf_dataset, test_indices)
    
    return train_subset, test_subset

def load_quickdraw_experiment(group_id, 
                              include_cf_in_train=True, 
                              cf_imgs=cf_imgs, 
                              true_labels=true_labels, batch_size=256):
    # Load original qd dataset (35k/5k)
    original_train_set = QuickDrawDataset(x_train_t, y_train_tensor, transform=train_transform)
    original_test_set = QuickDrawDataset(x_test_t, y_test_tensor, transform=test_transform)

    # Create two base CF datasets with different transforms (one for training with augmentation, one for testing without augmentation)
    cf_base_train_type = QuickDrawCFDataset(cf_imgs=cf_imgs, true_labels=true_labels, transform=train_transform)
    cf_base_test_type = QuickDrawCFDataset(cf_imgs=cf_imgs, true_labels=true_labels, transform=test_transform)

    # Load pre-generated user indices for the specified group
    user_indices = np.load(f'data/quickdraw_cf/cf_train_indices_group_{group_id}.npy')

    ######### DEBUG #########
    #visualize_full_cf_dataset(cf_base_train_type)
    

    
    # Get splits
    # 30 training samples (with augmentation)
    cf_train_subset, _ = get_cf_splits(cf_base_train_type, user_indices)
    # 120 test samples (without augmentation)
    _, cf_test_subset = get_cf_splits(cf_base_test_type, user_indices)

    # Concatenate original training set with CF subset if needed
    if include_cf_in_train:
        final_train_set = ConcatDataset([original_train_set, cf_train_subset])
        print(f"Data loaded with CF Group {group_id}")
    else:
        final_train_set = original_train_set
        print("Data loaded with Original Dataset Only")

    # Get data loaders
    train_loader = DataLoader(final_train_set, batch_size=batch_size, shuffle=True)
    orig_test_loader = DataLoader(original_test_set, batch_size=batch_size, shuffle=False)
    cf_test_loader = DataLoader(cf_test_subset, batch_size=batch_size, shuffle=False)
    
    full_cf_loader = DataLoader(cf_base_test_type, batch_size=batch_size, shuffle=False)

    return train_loader, orig_test_loader, cf_test_loader, full_cf_loader, user_indices


def load_quickdraw_data(batch_size=256):
    train_dataset = QuickDrawDataset(x_train_t, y_train_tensor, transform=train_transform)
    test_dataset = QuickDrawDataset(x_test_t, y_test_tensor, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


########## DEBUG FUNCTIONS ##########
def visualize_full_cf_dataset(cf_dataset, save_path="FULL_CF_DATASET_CHECK.png"):
    """
    Visualize the entire CF dataset in a grid format, where
    each row --> one of the 30 samples,
    each column --> one of the 5 users
    """
    
    fig, axes = plt.subplots(30, 5, figsize=(15, 60)) 
    
    print(f"Visualizing full CF dataset.. Total images: {len(cf_dataset)}. Expected: 150 (30 samples x 5 users)")

    for s_idx in range(30):
        for u_idx in range(5):
            global_idx = s_idx * 5 + u_idx
            
            img, label = cf_dataset[global_idx]
            
            ax = axes[s_idx, u_idx]
            
            ax.imshow(img.squeeze(), cmap='gray')
            
            title_text = f"S{s_idx} U{u_idx}\nL{int(label)}:{QD_CLASS_NAMES.get(int(label), '?')}"
            ax.set_title(title_text, fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualization saved in: {save_path}")