
 
import torch  
from torch import nn 
from architectures.UNET_CT.bottleneck import BottleNeck
from architectures.UNET_CT.conv_group import ConvGroup 

from architectures.UNET_CT.noise_level_embedding import NoiseLevelEmbedding
from architectures.UNET_CT.positional_embedding import PositionalEmbedding 

from architectures.UNET_CT.utils import POSITIONAL_TYPE


class LATENT_UNET(nn.Module):
     
  def __init__(self,
        device, 
        #1,2,2,4
               img_channels=3,mult=[1,2,4,8],base_channels=64,
        num_res_blocks = 3,
        num_head_channels=-1,
        num_heads=8,
        dropout=0.02,   
        num_classes=10,
        emb_type=POSITIONAL_TYPE,
        attention_resolution = [4,8,16],
        groupnorm=32,
        use_conv_in_res=False,
        use_conv_up=False,
        use_conv_down=False, 
        use_new_attention_order=False,
         use_flash_attention=False,  
        use_scale_shift_norm=False):
        
        super().__init__()        
        #self.base_channels=base_channels* 4
        #time_emb_dim = base_channels*mult[-2]
        time_emb_dim = base_channels*4
        if emb_type==POSITIONAL_TYPE: 
          self.time_mlp =   PositionalEmbedding(dim=time_emb_dim,device=device )
        else:
          self.time_mlp = NoiseLevelEmbedding(  channels=time_emb_dim,scale=0.02  )
          

        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        self.conv_in = nn.Conv2d(img_channels, base_channels* mult[0],kernel_size=3,padding= 1) 
        #self.conv_in =  nn.Sequential(  nn.Conv2d( img_channels, base_channels* mult[0]//2, 3, padding=1), nn.GroupNorm(groupnorm, base_channels* mult[0]//2),  nn.SiLU()  )
        
        resolution=1
        self.dconv_down1 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention, 
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res,  
                                        use_new_attention_order=use_new_attention_order, emb_type=emb_type )  
 
  
        resolution=2
        self.dconv_down2 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                      use_new_attention_order=use_new_attention_order,   emb_type=emb_type,
                                     num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads , use_conv_in_res=use_conv_in_res)
 
        resolution=4
        self.dconv_down3 = ConvGroup(in_channels=base_channels* mult[1],out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_new_attention_order=use_new_attention_order,   emb_type=emb_type,
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res) 
        


        resolution=8
        self.dconv_down4 = ConvGroup(in_channels=base_channels* mult[2],out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_new_attention_order=use_new_attention_order,   emb_type=emb_type,
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res) 
 
        resolution=16
        self.bottle_neck = BottleNeck(in_channels=base_channels* mult[3],out_channels=base_channels* mult[3],attention_resolution=attention_resolution,emb_channels=time_emb_dim,dropout=dropout, use_flash_attention=use_flash_attention,
                                      use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  resolution=resolution,  use_new_attention_order=use_new_attention_order,  
                                      num_head_channels=num_head_channels, num_heads=num_heads, use_conv_in_res=use_conv_in_res, emb_type=emb_type)  
 
   
        resolution=8
        self.dconv_up4 = ConvGroup(in_channels=   base_channels* mult[3]*2,out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                   use_new_attention_order=use_new_attention_order,   emb_type=emb_type,
                                       num_head_channels=num_head_channels, resolution=resolution , num_heads=num_heads,  use_conv_in_res=use_conv_in_res)  
         
        resolution=4
        self.dconv_up3 = ConvGroup(in_channels=   base_channels* mult[3] + base_channels* mult[2]  ,out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                   use_new_attention_order=use_new_attention_order,   emb_type=emb_type,
                                       num_head_channels=num_head_channels, resolution=resolution , num_heads=num_heads,  use_conv_in_res=use_conv_in_res)  
         
          
        resolution=2
        self.dconv_up2 =  ConvGroup(in_channels=  base_channels* mult[2]+ base_channels* mult[1],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,    emb_type=emb_type,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,  use_conv_in_res=use_conv_in_res )   
                
        resolution=1
        self.dconv_up1 =  ConvGroup(in_channels= base_channels* mult[1] + base_channels* mult[0],out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,    emb_type=emb_type,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, use_conv_in_res=use_conv_in_res)
        
        #self.conv_out= nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1 ) 
        
        
        self.conv_out = nn.Sequential( nn.GroupNorm(groupnorm,base_channels* mult[0]), nn.SiLU(),   
                                   nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1 ))
  def forward(self, x, time, cond=None ):    
 
 
        time_emb = self.time_mlp(time)
        if cond is not None: 
             time_emb+=self.label_emb(cond)
        #
        x= self.conv_in(x) 

        conv1 = self.dconv_down1(x,time_emb)   
        x=conv1
        conv2 = self.dconv_down2(x,time_emb) 
        x=conv2
        
        conv3 = self.dconv_down3(x,time_emb)  

        x=conv3
        
        conv4 = self.dconv_down4(x,time_emb)  
        x=conv4

        x = self.bottle_neck(x,time_emb)  
 
        x = self.dconv_up4(torch.cat([x, conv4], dim=1)  ,time_emb)  

 
        x = self.dconv_up3(torch.cat([x, conv3], dim=1) ,time_emb)
         
        x = self.dconv_up2(torch.cat([x, conv2], dim=1) ,time_emb) 
 
        x = self.dconv_up1(torch.cat([x, conv1], dim=1),time_emb ) 
        
         
        x = self.conv_out(x) 
        return x
