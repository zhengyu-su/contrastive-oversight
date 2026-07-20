import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickDrawCNN(nn.Module):
    def __init__(self):
        super(QuickDrawCNN, self).__init__()
        # Keras Conv2D 默认 kernel_size=(3,3) 才能得到 summary 里的参数量 (160)
        # 160 = (3*3 * 1 * 32) + 32 bias
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)

        # 8256 = (3*3 * 32 * 64) + 64 bias
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.3)

        # 28x28 -> pool -> 14x14 -> pool -> 7x7.  64*7*7 = 3136
        self.fc1 = nn.Linear(3136, 256)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)    

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x