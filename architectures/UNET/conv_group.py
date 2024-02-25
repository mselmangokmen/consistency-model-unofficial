
 
import torch.nn as nn

from architectures.UNET.conv_block import ConvBlock
from architectures.UNET.resblock import ResBlock
 


 
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
        attention_resolution = [4,8,16],
        use_scale_shift_norm=False,
        resolution=1,
        up=False,
        down=False,
        use_conv=False,
        use_new_attention_order=False,
        use_flash_attention=False

    ):
        super().__init__()
        layers = [] 
        layers.append(ConvBlock(in_channels=in_channels,attention_resolution=attention_resolution,dropout=dropout,emb_channels=emb_channels,
                                    out_channels=out_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm, use_flash_attention=use_flash_attention,
                                    num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv=use_conv, use_new_attention_order=use_new_attention_order ))
        
        for num in range(num_res_blocks-1): 
                layers.append(ConvBlock(in_channels=out_channels,attention_resolution=attention_resolution,dropout=dropout,emb_channels=emb_channels,
                                    out_channels=out_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm, use_flash_attention=use_flash_attention,
                                    num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv=use_conv,use_new_attention_order=use_new_attention_order ))

        resblock= ResBlock(in_channels=out_channels,out_channels=out_channels,dropout=dropout,emb_channels=emb_channels,
                           use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm, up=up,down=down, use_conv=use_conv)
        layers.append(resblock)
        self.seq= TimestepEmbedSequential(*layers)
    def forward(self, x, emb):

        return self.seq(x,emb) 
