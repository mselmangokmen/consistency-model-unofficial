
 
import torch.nn as nn  
import torch
from architectures.UNET.downsample import Downsample

from architectures.UNET.upsample import Upsample 
 
class ResBlock(nn.Module):
    

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels,
        dropout=0.02,
        groupnorm= 16,
        use_scale_shift_norm=False,
        use_conv=False,
        use_conv_up_down=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels 

        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(groupnorm, in_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d( in_channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(groupnorm,self.out_channels),
            nn.SiLU(inplace=False),
            nn.Dropout(p=dropout), 
              nn.Conv2d( self.out_channels, self.out_channels, 3, padding=1)
                
         
        )
 

        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, use_conv_up_down)
            self.x_upd = Upsample(in_channels, use_conv_up_down)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv_up_down)
            self.x_upd = Downsample(in_channels, use_conv_up_down)
        else:
            self.h_upd = self.x_upd = nn.Identity()
 

        if use_conv:
            self.skip_connection =   nn.Conv2d(  in_channels, self.out_channels, 3, padding=1) 
        else:
            self.skip_connection = nn.Conv2d(  in_channels, self.out_channels, 1) 


    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        #print(emb_out.shape)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
