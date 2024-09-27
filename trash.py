import torch.nn as nn
import torch

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Change input channels from 1 to 3 (since we now have RGB input)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Adjust the fully connected layer size based on the output of the convolutions
        self.fc1 = nn.Linear(32 * 7 * 7, 750)
        self.fc2 = nn.Linear(750, 4)  # 4 outputs for 4 actions (up, down, left, right)
        
        self.gelu = nn.GELU()  # Activation function

    def forward(self, x):
        # x shape [batch, 3, 30, 30]
        print('Input shape:', x.shape)  # Check input shape
        
        # Apply first convolution + activation + pooling
        x = self.pool(self.gelu(self.conv1(x)))  # Shape: [batch, 16, 15, 15]
        
        # Apply second convolution + activation + pooling
        x = self.pool(self.gelu(self.conv2(x)))  # Shape: [batch, 32, 7, 7]
        
        # Flatten the tensor to a vector
        x = x.view(x.size(0), -1)  # Shape: [batch, 32 * 7 * 7] = [batch, 1568]
        
        # Fully connected layer + activation
        x = self.gelu(self.fc1(x))  # Shape: [batch, 750]
        
        # Output layer (Q-values for each action)
        x = self.fc2(x)  # Shape: [batch, 4] (for 4 actions)
        
        return x  # Return Q-values for each action

