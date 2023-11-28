



import torch.nn.functional as F
        
from torch import nn
class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm,time_emb_dim):
 
        super().__init__() 
        kernel_size=3 
        self.time_bias = nn.Linear(time_emb_dim, out_channels) 
 
        self.activation= F.relu


        self.seq_1 =  nn.Sequential(
            
             nn.GroupNorm(group_norm, num_channels=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
    )  
        self.res_con = nn.Conv2d(in_channels, out_channels, 1) 
        
         
    def forward(self, x,time_emb=None):
        res= x
        time_emb= self.time_bias(self.activation(time_emb))[:, :, None, None]
        out = self.seq_1(x)+ time_emb 

        return out + self.res_con(res)