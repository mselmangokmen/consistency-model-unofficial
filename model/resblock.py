
 
import torch.nn as nn  
import torch  
 
class ResBlock(nn.Module):
    

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels,
        dropout=0.02,
        groupnorm= 32,
        use_scale_shift_norm=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels 

        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(groupnorm, in_channels),
            nn.SiLU(),
            nn.Conv2d( in_channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(groupnorm,self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout), 
                nn.Conv2d( self.out_channels, self.out_channels, 3, padding=1)
         
        )

        self.skip_connection = nn.Conv2d(  in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        
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
