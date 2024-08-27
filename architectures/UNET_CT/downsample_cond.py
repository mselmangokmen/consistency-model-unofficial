
 
import torch.nn as nn 
import torch.nn.functional as F

 
 
class Downsample_Cond(nn.Module):
    
    def __init__(self, channels, 
        emb_channels  ,stride=2,kernel=3,padding=1, groupnorm=32):
        super().__init__()
        self.channels = channels    
        self.op = nn.Conv2d(   self.channels, self.channels, kernel_size=kernel, stride=stride, padding=padding  )
        self.op2 = nn.Sequential( nn.GroupNorm(groupnorm,self.channels),  nn.SiLU(), 
                                           nn.Conv2d( self.channels, self.channels, 3, padding=1)  )
  
    def forward(self, x, emb=None):       
        x = self.op(x) 
        x = self.op2(x)
        return x
