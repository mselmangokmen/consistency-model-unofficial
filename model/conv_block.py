
 
import torch.nn as nn 
import torch.nn.functional as F
from model.attention_block import AttentionBlock  
from model.resblock import ResBlock 

  
class ConvBlock(nn.Module):
    

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        dropout=0.02, 
        groupnorm=32,
        attention_resolution = 8,
        use_scale_shift_norm=False,

    ):
        super().__init__()
        self.resblock= ResBlock(in_channels=in_channels,out_channels=out_channels,dropout=dropout,emb_channels=emb_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)
        self.self_att=  AttentionBlock(channels=out_channels,num_heads=attention_resolution)  
    def forward(self, x, emb):
        x = self.resblock(x,emb)
        x= self.self_att(x)
        return x
