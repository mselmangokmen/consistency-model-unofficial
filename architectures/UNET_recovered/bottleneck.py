
 
import torch.nn as nn
from architectures.UNET_recovered.attention_block import AttentionBlock
from architectures.UNET_recovered.normal_attention import NormalAttentionBlock

from architectures.UNET_recovered.resblock import ResBlock 

 
class BottleNeck(nn.Module):
    

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.02, 
        groupnorm=32,
        attention_resolution = 8,
        num_head_channels=-1,
         num_heads=8,
        use_scale_shift_norm=False,
         use_new_attention_order=True,
         use_flash_attention=False, 
        resolution=1,  
         use_conv_in_res=True

    ):
        super().__init__()
        self.res= resolution 
        self.resblock_1= ResBlock(in_channels=channels,out_channels=channels,dropout=dropout,  
                                  emb_channels=emb_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm, use_conv_in_res=use_conv_in_res)
        self.self_att=None
        if resolution in attention_resolution:
            print(resolution)
            if use_flash_attention:
                self.self_att=  AttentionBlock(channels=channels,num_heads=num_heads,groupnorm_ch=groupnorm, num_head_channels=num_head_channels)  
            else:
                self.self_att=  NormalAttentionBlock(channels=channels,num_heads=num_heads,groupnorm_ch=groupnorm, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order) 
        self.resblock_2= ResBlock(in_channels=channels,out_channels=channels,dropout=dropout,   
                                  emb_channels=emb_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm, use_conv_in_res=use_conv_in_res)
    def forward(self, x, emb):
        x = self.resblock_1(x,emb)
        if self.self_att:
            #print('self_att ok . res: '+ str(self.res))
            #print('attention used,  resolution : '+ str(self.resolution))
            x= self.self_att(x)
        x = self.resblock_2(x,emb)
        return x
