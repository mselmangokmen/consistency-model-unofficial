



import torch.nn.functional as F
        
from torch import nn
import torch
class UpDownSampling(nn.Module):
    def __init__(self, channels, group_norm,time_emb_dim,sampling='up'):
 
        super().__init__() 
        kernel_size=1
        padding=0
        self.time_bias = nn.Linear(time_emb_dim, channels)  
        self.sampling =  nn.Upsample(scale_factor=2 ) if sampling=='up' else nn.MaxPool2d(2)
        self.activation= F.relu 
        self.seq_1 =  nn.Sequential(  
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding), 
             nn.GroupNorm(group_norm, num_channels=channels),
            nn.SiLU()
    )
        

    def forward(self, x,time_emb=None):
         
        time_emb= self.time_bias(self.activation(time_emb))[:, :, None, None]
        res= x.clone()
        x= self.seq_1(x) 
        out = torch.add(x,res )

        out = self.sampling(out)
        out = out+ time_emb 
        return out 