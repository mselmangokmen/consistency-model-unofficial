
 
import torch.nn as nn
from architectures.UNET.attention_block import AttentionBlock 

from architectures.UNET.resblock import ResBlock
from architectures.UNET.utils import POSITIONAL_TYPE 

 
class BottleNeck(nn.Module):
    

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        dropout=0.02, 
        groupnorm=32, 
        num_head_channels=-1,
         num_heads=8,  
        resolution=1,   

    ):
        super().__init__()
        self.res= resolution 
        self.resblock_1= ResBlock(in_channels=in_channels,out_channels=out_channels,dropout=dropout,  
                                  emb_channels=emb_channels, groupnorm=groupnorm )
 
        self.self_att=  AttentionBlock(channels=out_channels,num_heads=num_heads,groupnorm_ch=groupnorm, num_head_channels=num_head_channels)  
        
        self.resblock_2= ResBlock(in_channels=out_channels,out_channels=out_channels,dropout=dropout,   
                                  emb_channels=emb_channels, groupnorm=groupnorm )
    def forward(self, x, emb):
        x = self.resblock_1(x,emb) 
        x= self.self_att(x)
        x = self.resblock_2(x,emb)
        return x
