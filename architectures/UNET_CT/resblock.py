
 
import torch.nn as nn   
  
class ResBlock(nn.Module):
    

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels,
        dropout=0.02,
        groupnorm= 16,   
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels  
        self.in_layers = nn.Sequential(  nn.GroupNorm(groupnorm, in_channels),  nn.SiLU(inplace=True), nn.Conv2d( in_channels, self.out_channels, 3, padding=1) )
        self.out_layers = nn.Sequential( nn.GroupNorm(groupnorm,self.out_channels),  nn.SiLU(),nn.Dropout(p=dropout),
                                           nn.Conv2d( self.out_channels, self.out_channels, 3, padding=1)  )

 
        self.emb_layers = nn.Sequential(
                nn.SiLU(), 
                nn.Conv2d(emb_channels, out_channels, kernel_size=1), 
            )
 
 
        self.skip_connection = nn.Conv2d(  in_channels, self.out_channels, 1) 


    def forward(self, x, emb=None): 
        h = self.in_layers(x)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
        else: 
            emb_out=nn.Identity()
            
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h
