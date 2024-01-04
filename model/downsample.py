
 
import torch.nn as nn 
import torch.nn.functional as F

 

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels ):
        super().__init__()
        self.channels = channels  
        stride = 2  
        self.op = nn.Conv2d(   self.channels, self.channels, 3, stride=stride, padding=1  )

    def forward(self, x): 
        return self.op(x)
