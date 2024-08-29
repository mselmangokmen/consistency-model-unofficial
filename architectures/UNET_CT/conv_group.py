
import torch.nn as nn
  
from architectures.UNET_CT.conv_block import ConvBlock
from architectures.UNET_CT.utils import POSITIONAL_TYPE 
   


 
 
class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb)
        return x



class ConvGroup(nn.Module):
    

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        num_res_blocks = 6,
        dropout=0.02, 
        groupnorm=16,
        num_head_channels=-1,
        num_heads=8,
         
        attention_resolution = [8,16],
 
        resolution=1,    
    ):
        
        super().__init__()
        layers = [] 
        in_channel_size=in_channels     

        for num in range(num_res_blocks):  
            
                layers.append(ConvBlock(in_channels=in_channel_size,attention_resolution=attention_resolution,dropout=dropout,emb_channels=emb_channels, 
                                        out_channels=out_channels, groupnorm=groupnorm,  
                                        num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,  
                                       ))
                in_channel_size= out_channels
                  
        self.seq= TimestepEmbedSequential(*layers)
    def forward(self, x, emb):
        
        return self.seq(x,emb)  

 