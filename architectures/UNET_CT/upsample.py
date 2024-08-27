 
 
import torch.nn as nn 
import torch.nn.functional as F
 
 

class Upsample(nn.Module): 
    def __init__(self, channels ):
        super().__init__()
        self.channels = channels  
        self.conv = nn.ConvTranspose2d(
                self.channels, 
                self.channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                output_padding=0
            ) 
            
    def forward(self, x):  
        x = self.conv(x) 
        return x
