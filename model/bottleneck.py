
 
import torch.nn as nn  
from model.openai.attention_block import AttentionBlock 
from model.openai.resblock import ResBlock 

 
class BottleNeck(nn.Module):
    

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.02, 
        groupnorm=32,
        attention_resolution = 8,
        use_scale_shift_norm=False,

    ):
        super().__init__()
        self.resblock_1= ResBlock(in_channels=channels,out_channels=channels,dropout=dropout,emb_channels=emb_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)
        self.self_att=  AttentionBlock(channels=channels,num_heads=attention_resolution)  
        self.resblock_2= ResBlock(in_channels=channels,out_channels=channels,dropout=dropout,emb_channels=emb_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)
    def forward(self, x, emb):
        x = self.resblock_1(x,emb)
        x= self.self_att(x)
        x = self.resblock_2(x,emb)
        return x
