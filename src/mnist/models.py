'''
torch model for mnist dataset according to the original keras model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class mnistCNN(nn.Module):
    def __init__(self):
        super(mnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3) # not indicated in keras summary
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3) # not indicated in keras summary
    
        self.fc1 = nn.Linear(3136, 256)
        self.dropout3 = nn.Dropout(0.5) # not indicated in keras summary
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1)) 
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # keras: N,H,W,C，PyTorch: N,C,H,W
        # Adjust dimensions for flattening
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def get_mnist_model(model_path=None, device="cpu"):
    """
    Function to initialize the model (and load weights if provided)
    """
    model = mnistCNN().to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # load the model for inference in evaluation mode
    return model