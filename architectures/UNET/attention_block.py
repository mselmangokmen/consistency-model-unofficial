from torch import nn 

from architectures.UNET.flash_attention import FlashAttention
 
 
  
 

class AttentionBlock(nn.Module):
    

    def __init__(
        self,
        channels,
        num_heads=8,
        num_head_channels=-1,
        groupnorm_ch=16
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels==-1:
            self.num_heads=num_heads
        else:
            self.num_heads = channels // num_head_channels
            print('channels: ',channels)
            print('num_head_channels: ',num_head_channels)
            print('num_heads: ',self.num_heads)
        self.groupnorm =  nn.GroupNorm(groupnorm_ch, channels)
        self.res_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_out =  nn.Conv2d(channels, channels, kernel_size=1, padding=0)  
        self.layernorm_1 = nn.LayerNorm(channels)

        self.layernorm_2 = nn.LayerNorm(channels)
        
        self.d_head = channels // self.num_heads
        print('d_head: ',self.d_head)
        self.attention =  FlashAttention(dim=channels, heads=self.num_heads, dim_head=self.d_head)

    def forward(self, x):
 
        residue = x.clone()
        residue= self.res_input(residue)
        residue = self.groupnorm(residue) 
 
        x = self.groupnorm(x) 
        x = self.conv_input(x) 
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2).contiguous()
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)        
        x = x.transpose(-1, -2).contiguous()
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        x= self.conv_out(x)
        x= x + residue 
        return x

  