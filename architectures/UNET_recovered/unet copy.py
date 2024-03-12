
 
import torch  
from torch import nn
from architectures.UNET_recovered.bottleneck import BottleNeck
from architectures.UNET_recovered.conv_group import ConvGroup
from architectures.UNET_recovered.downsample import Downsample

from architectures.UNET_recovered.positional_embedding import PositionalEmbedding
from architectures.UNET_recovered.resblock import ResBlock
from architectures.UNET_recovered.upsample import Upsample 
from architectures.UNET_recovered.utils import zero_module 



class UNET_recovered(nn.Module):
    


  def __init__(self,device, 
               img_channels=3,mult=[1,2,4,8],base_channels=64,
               time_emb_scale=1,
        num_res_blocks = 3,
        num_head_channels=-1,
        num_heads=6,
        dropout=0.02, 
        attention_resolution = [4,8,16],
        groupnorm=32,
        use_conv_in_res=False,
        use_conv_up_down=False,
        use_new_attention_order=False,
         use_flash_attention=False,
         reverse=False,
        use_scale_shift_norm=False):
        
        super().__init__()        

        #time_emb_dim= 256
        #self.time_mlp =    PositionalEmbedding(dim=time_emb_dim, scale=time_emb_scale,device=device)
        #max_level = max(mult)

        time_emb_dim = base_channels*4
        #time_emb_dim=base_channels *max_level 
        #if reverse:
        self.reverse= reverse
        self.time_mlp = nn.Sequential( PositionalEmbedding(dim=base_channels, scale=time_emb_scale,device=device),nn.Linear(base_channels, time_emb_dim),nn.SiLU(inplace=False),nn.Linear(time_emb_dim, time_emb_dim) )
        #else:
        #self.time_mlp = nn.Sequential( PositionalEmbedding(dim=base_channels, scale=time_emb_scale,device=device),nn.Linear(base_channels, time_emb_dim),nn.SiLU())

        #self.conv_input = nn.Sequential(  nn.Conv2d(3, (base_channels* mult[0]),kernel_size=3,padding=1)  ,   nn.GroupNorm(groupnorm, (base_channels* mult[0])),  nn.SiLU() )
        if reverse:
          self.conv_in = nn.Conv2d(3, base_channels* mult[0]//2,kernel_size=3,padding=1) 

        else: 
          self.conv_in = ResBlock(in_channels=img_channels,out_channels=base_channels* mult[0]//2,dropout=dropout,emb_channels=time_emb_dim, use_conv_up_down=use_conv_up_down, reverse=reverse,
                              use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_conv_in_res=use_conv_in_res)
          
        resolution=1
        self.dconv_down1 = ConvGroup(in_channels=base_channels* mult[0]//2,out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention, 
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res, reverse=reverse,
                                       use_conv_up_down=use_conv_up_down,down=True,use_new_attention_order=use_new_attention_order )  
  
        resolution=2
        self.dconv_down2 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                         use_conv_up_down=use_conv_up_down,down=True, use_new_attention_order=use_new_attention_order, reverse=reverse,
                                     num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads , use_conv_in_res=use_conv_in_res)
 
         
        
        resolution=4
        self.dconv_down3 = ConvGroup(in_channels=base_channels* mult[1],out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_conv_up_down=use_conv_up_down,down=True, use_new_attention_order=use_new_attention_order, reverse=reverse,
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res) 
         
   
        resolution=8
        self.dconv_down4 = ConvGroup(in_channels=base_channels* mult[2],out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_conv_up_down=use_conv_up_down,down=True, use_new_attention_order=use_new_attention_order, reverse=reverse,
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res) 
 
           

        resolution=16
        self.bottle_neck = BottleNeck(channels=base_channels* mult[3],attention_resolution=attention_resolution,emb_channels=time_emb_dim,dropout=dropout, use_flash_attention=use_flash_attention,
                                      use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  resolution=resolution,  use_new_attention_order=use_new_attention_order, reverse=reverse,
                                      num_head_channels=num_head_channels, num_heads=num_heads, use_conv_in_res=use_conv_in_res)  

          

        resolution=8
        self.dconv_up4 = ConvGroup(in_channels=base_channels* mult[3] *2,out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                      use_conv_up_down=use_conv_up_down,up=True,use_new_attention_order=use_new_attention_order,   reverse=reverse,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res )  
        

             
  
        resolution=4
        self.dconv_up3 = ConvGroup(in_channels=base_channels* mult[2] *2,out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_conv_up_down=use_conv_up_down,up=True,use_new_attention_order=use_new_attention_order,   reverse=reverse,
                                       num_head_channels=num_head_channels, resolution=resolution , num_heads=num_heads,  use_conv_in_res=use_conv_in_res)  
         
              
      
        resolution=2
        self.dconv_up2 =  ConvGroup(in_channels=base_channels* mult[1] *2,out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_conv_up_down=use_conv_up_down,up=True,use_new_attention_order=use_new_attention_order,  reverse=reverse,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res )   
               
 
        resolution=1
        self.dconv_up1 =  ConvGroup(in_channels=base_channels* mult[0] *2,out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_conv_up_down=use_conv_up_down,up=True,use_new_attention_order=use_new_attention_order,  reverse=reverse,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res)
        
        self.conv_out = ResBlock(in_channels=base_channels* mult[0],out_channels=base_channels* mult[0]//2,dropout=dropout,emb_channels=time_emb_dim, use_conv_up_down=use_conv_up_down, reverse=reverse,
                            use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_conv_in_res=use_conv_in_res)
        if reverse:
          self.conv_last=  nn.Sequential( nn.GroupNorm(groupnorm, base_channels* mult[0]//2),
               nn.SiLU(), 
          nn.Conv2d(base_channels* mult[0]//2, img_channels,kernel_size=3,padding=1 )) 
        else: 
          self.conv_last= nn.Conv2d(base_channels* mult[0]//2, img_channels,kernel_size=3,padding=1 )
        
        
          
 
  def forward(self, x, time ):    
 
        time_emb = self.time_mlp(time) 

        if self.reverse:
          x= self.conv_in(x) 
        else:
          x= self.conv_in(x,time_emb) 

        conv1 = self.dconv_down1(x,time_emb) 
        conv2 = self.dconv_down2(conv1,time_emb) 
        conv3 = self.dconv_down3(conv2,time_emb) 
        conv4 = self.dconv_down4(conv3,time_emb)  

        x = self.bottle_neck(conv4,time_emb)    

        x = self.dconv_up4(torch.cat([x, conv4], dim=1)  ,time_emb)    
        x = self.dconv_up3(torch.cat([x, conv3], dim=1) ,time_emb)  
        x = self.dconv_up2(torch.cat([x, conv2], dim=1) ,time_emb) 
        x = self.dconv_up1(torch.cat([x, conv1], dim=1),time_emb)  
        x = self.conv_out(x,time_emb)  
        x = self.conv_last(x)  
        return x
