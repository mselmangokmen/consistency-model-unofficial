
 
import torch.nn as nn

from model.conv_block import ConvBlock   



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
        num_res_blocks = 3,
        dropout=0.02, 
        groupnorm=32,
        attention_resolution = 8,
        use_scale_shift_norm=False,


    ):
        super().__init__()
        layers = []
        layers.append(ConvBlock(in_channels=in_channels,attention_resolution=attention_resolution,dropout=dropout,emb_channels=emb_channels,
                                    out_channels=out_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm))
        for num in range(num_res_blocks-1):
            layers.append(ConvBlock(in_channels=out_channels,attention_resolution=attention_resolution,dropout=dropout,emb_channels=emb_channels,
                                    out_channels=out_channels,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm))
        self.seq= TimestepEmbedSequential(*layers)
    def forward(self, x, emb):

        return self.seq(x,emb)
