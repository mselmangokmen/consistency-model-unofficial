
 
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
        use_scale_shift_norm=False):
        
        super().__init__()        

        #time_emb_dim= 256
        #self.time_mlp =    PositionalEmbedding(dim=time_emb_dim, scale=time_emb_scale,device=device)
        #max_level = max(mult)

        time_emb_dim = base_channels*4
        #time_emb_dim=base_channels *max_level 
        #if reverse: 
        self.time_mlp = nn.Sequential( PositionalEmbedding(dim=base_channels, scale=time_emb_scale,device=device),nn.Linear(base_channels, time_emb_dim),nn.SiLU(inplace=False),nn.Linear(time_emb_dim, time_emb_dim) )

        self.conv_in = nn.Conv2d(3, base_channels* mult[0]//2,kernel_size=3,padding=1) 

        resolution=1
        self.dconv_down1 = ConvGroup(in_channels=base_channels* mult[0]//2,out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention, 
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res,  
                                        use_new_attention_order=use_new_attention_order )  
        self.down_sample1=Downsample(channels=base_channels* mult[0], use_conv=use_conv_up_down)
  
        resolution=2
        self.dconv_down2 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                      use_new_attention_order=use_new_attention_order,  
                                     num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads , use_conv_in_res=use_conv_in_res)
 
         
        self.down_sample2=Downsample(channels=base_channels* mult[1], use_conv=use_conv_up_down)

        resolution=4
        self.dconv_down3 = ConvGroup(in_channels=base_channels* mult[1],out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_new_attention_order=use_new_attention_order,  
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res) 
         
   
        self.down_sample3=Downsample(channels=base_channels* mult[2], use_conv=use_conv_up_down)
        resolution=8
        self.dconv_down4 = ConvGroup(in_channels=base_channels* mult[2],out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                      use_new_attention_order=use_new_attention_order,  
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res) 
 
           

        self.down_sample4=Downsample(channels=base_channels* mult[3], use_conv=use_conv_up_down)
        resolution=16
        self.bottle_neck = BottleNeck(channels=base_channels* mult[3],attention_resolution=attention_resolution,emb_channels=time_emb_dim,dropout=dropout, use_flash_attention=use_flash_attention,
                                      use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  resolution=resolution,  use_new_attention_order=use_new_attention_order,  
                                      num_head_channels=num_head_channels, num_heads=num_heads, use_conv_in_res=use_conv_in_res)  

        

        self.up_sample4=Upsample(channels=base_channels* mult[3], use_conv=use_conv_up_down)
        resolution=8
        self.dconv_up4 = ConvGroup(in_channels=base_channels* mult[3] *2,out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,   
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res )  
        

         
  
        self.up_sample3=Upsample(channels=base_channels* mult[2], use_conv=use_conv_up_down)
        resolution=4
        self.dconv_up3 = ConvGroup(in_channels=base_channels* mult[2] *2,out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                   use_new_attention_order=use_new_attention_order,  
                                       num_head_channels=num_head_channels, resolution=resolution , num_heads=num_heads,  use_conv_in_res=use_conv_in_res)  
         
              
      
        self.up_sample2=Upsample(channels=base_channels* mult[1], use_conv=use_conv_up_down)
        resolution=2
        self.dconv_up2 =  ConvGroup(in_channels=base_channels* mult[1] *2,out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,   
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res )   
               
 
        self.up_sample1=Upsample(channels=base_channels* mult[0], use_conv=use_conv_up_down)
        resolution=1
        self.dconv_up1 =  ConvGroup(in_channels=base_channels* mult[0] *2,out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,   
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res)
        
        self.conv_last= nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1 ) 
 
  def forward(self, x, time ):    
 

        time_emb = self.time_mlp(time)
        #
        x= self.conv_in(x) 

        conv1 = self.dconv_down1(x,time_emb)  
        x = self.down_sample1(conv1)
        
        conv2 = self.dconv_down2(x,time_emb)
        x = self.down_sample2(conv2)
        
        
        conv3 = self.dconv_down3(x,time_emb) 
        x = self.down_sample3(conv3)   
        
        conv4 = self.dconv_down4(x,time_emb) 
        x = self.down_sample4(conv4)  

        x = self.bottle_neck(x,time_emb)  

        x = self.up_sample4(x)      
        x = self.dconv_up4(torch.cat([x, conv4], dim=1)  ,time_emb)  


        x = self.up_sample3(x)         
        x = self.dconv_up3(torch.cat([x, conv3], dim=1) ,time_emb)
        
        x = self.up_sample2(x)          
        x = self.dconv_up2(torch.cat([x, conv2], dim=1) ,time_emb) 


        x = self.up_sample1(x)         
        x = self.dconv_up1(torch.cat([x, conv1], dim=1),time_emb ) 
        
        
        x = self.conv_last(x) 
        return x
