import torch as th 
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

from architectures.UNET_recovered.flash_attention import FlashAttention

from architectures.UNET_recovered.utils import zero_module
 
  
 

class AttentionBlock(nn.Module):
    

    def __init__(
        self,
        channels,
        num_heads=8,
        num_head_channels=-1,
        groupnorm_ch=16
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels==-1:
            self.num_heads=num_heads
        else:
            self.num_heads = channels // num_head_channels
        self.groupnorm =  nn.GroupNorm(groupnorm_ch, channels)
        self.res_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_out =  nn.Conv2d(channels, channels, kernel_size=1, padding=0) 
        self.layernorm_1 = nn.LayerNorm(channels)

        self.layernorm_2 = nn.LayerNorm(channels)
        
        self.d_head = channels // num_heads
        self.attention =  FlashAttention(dim=channels, heads=num_heads, dim_head=self.d_head)

    def forward(self, x):
 
        residue = x.clone()
        residue= self.res_input(residue)
        residue = self.groupnorm(residue) 
 
        x = self.groupnorm(x) 
        x = self.conv_input(x) 
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2).contiguous()
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)        
        x = x.transpose(-1, -2).contiguous()
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        x= self.conv_out(x)
        return x + residue

  