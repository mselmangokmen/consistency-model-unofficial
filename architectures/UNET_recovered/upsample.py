 
 
import torch.nn as nn 
import torch.nn.functional as F
 
 

class Upsample(nn.Module): 
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels 
        self.use_conv = use_conv 
        if use_conv:
            self.conv = nn.Conv2d( self.channels, self.channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
