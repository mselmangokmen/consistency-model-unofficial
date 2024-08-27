import torch
import torch.nn as nn
import torch.nn.functional as F

class BigGANResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, 
                  activation=nn.ReLU, 
                 norm_layer=nn.BatchNorm2d):
        super(BigGANResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels else out_channels 
        
        self.activation = activation()
        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(self.hidden_channels)
        
        self.conv1 = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=3, padding=1)
         
        if in_channels != out_channels :
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = None
        
    def forward(self, x):
        residual = x
         
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
         
        if self.shortcut:
            residual = self.shortcut(residual)
        
        return x + residual
 