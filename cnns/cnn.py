import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels=1, output_size=1, init_type='he', conv_activation_function= F.sigmoid, fc_activation_function= F.sigmoid):
        super(CNN, self).__init__()
        
        # --- Convolutional layers ---
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # GAP sort un vecteur de taille 128
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # --- Fully connected layers ---
        self.fc1 = nn.Linear(128, 256)  # For 64x64 input
        self.fc2 = nn.Linear(256, output_size)
        
        # --- Weight initialization ---
        self.init_weights(init_type)
        
        # --- Activation functions ---
        self.conv_activation_function = conv_activation_function
        self.fc_activation_function = fc_activation_function

    def init_weights(self, init_type='he'):
        """Initialize weights according to chosen strategy"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'he':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == 'uniform':
                    nn.init.uniform_(m.weight, -0.05, 0.05)
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, 0.02)
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, conv_activation_function=F.sigmoid, fc_activation_function=F.sigmoid):
        # --- Convolutional layers ---
        x = conv_activation_function(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = conv_activation_function(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = conv_activation_function(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = self.gap(x)              # (B, 128, 1, 1)
        
        # --- Flatten ---
        x = x.view(x.size(0), -1)
        
        # --- Fully connected layers ---
        x = fc_activation_function(self.fc1(x))
        x = self.fc2(x)  # linear output for regression
        
        return x
