



import torch.nn.functional as F
        
from torch import nn 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm,time_emb_dim):
 
        super().__init__() 
        kernel_size=3 
        self.time_bias = nn.Linear(time_emb_dim, out_channels) 
        mid_channels = (in_channels+ out_channels)//2
        self.activation= F.relu 
        self.seq_1 =  nn.Sequential( 

            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1),
             nn.GroupNorm(group_norm, num_channels=out_channels),
            nn.SiLU()
    )   
        
         
    def forward(self, x,time_emb=None):
         
        time_emb= self.time_bias(self.activation(time_emb))[:, :, None, None]
        
        x= self.seq_1(x)
        out = x+ time_emb  
        return out 