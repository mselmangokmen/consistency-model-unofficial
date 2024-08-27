
 
import torch.nn as nn
from architectures.UNET_CT.attention_block import AttentionBlock  
 
from architectures.UNET_CT.normal_attention import NormalAttentionBlock
from architectures.UNET_CT.resblock import ResBlock
from architectures.UNET_CT.utils import POSITIONAL_TYPE 
 
class ConvBlock(nn.Module):
    

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        dropout=0.02, 
        groupnorm=16,
        num_head_channels=-1,
        attention_resolution = [ 8,16],
        num_heads=8, 
         resolution=1  ):
        super().__init__() 
        self.res= resolution
        self.self_att=None
        self.resblock= ResBlock(in_channels=in_channels,out_channels=out_channels,dropout=dropout,emb_channels=emb_channels, 
                                groupnorm=groupnorm   )
        if resolution in attention_resolution: 
            print('res: '+ str(resolution)) 
            self.self_att=  NormalAttentionBlock(channels=out_channels,num_heads=num_heads,groupnorm_ch=groupnorm, num_head_channels=num_head_channels)  
 
        
            print(resolution)
    def forward(self, x, emb):
        x = self.resblock(x,emb)
        if self.self_att: 
            x= self.self_att(x)
        return x


 