
 
import torch  
from torch import nn
from architectures.UNET.bottleneck import BottleNeck
from architectures.UNET.conv_group import ConvGroup
from architectures.UNET.downsample import Downsample

from architectures.UNET.positional_embedding import PositionalEmbedding
from architectures.UNET.upsample import Upsample
from architectures.UNET.utils import zero_module    



class UNET(nn.Module):
    


  def __init__(self,device, eps= 0.002, 
               img_channels=3,mult=[1,2,4,8],base_channels=64,
               time_emb_scale=1,
        num_res_blocks = 3,
        num_head_channels=-1,
        num_heads=6,
        dropout=0.02, 
        attention_resolution = [4,8,16],
        groupnorm=32,
        use_conv=False,
        use_new_attention_order=False,
         use_flash_attention=False,
        use_scale_shift_norm=False):
        
        super().__init__()   
        self.eps = eps
        self.dilation=1 
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=False)
        self.time_emb_dim=base_channels*4
        #self.upsample = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)    
        
        self.sigmoid = nn.Sigmoid() 
        self.encoder_layers=[]
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(dim=self.time_emb_dim, scale=time_emb_scale,device=device),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(inplace=False),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
 
        self.conv_input = nn.Conv2d(3, base_channels* mult[0],kernel_size=3,padding=1)  
        #self.conv_input = nn.Sequential(  nn.Conv2d(3, (base_channels* mult[0])//2,kernel_size=3,padding=1)  ,    nn.GroupNorm(groupnorm, (base_channels* mult[0])//2),  nn.SiLU() )
        resolution= 1
        self.dconv_down1 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads, down=True, use_conv=use_conv, use_new_attention_order=use_new_attention_order )  
  
        
        resolution=2
        self.dconv_down2 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_new_attention_order=use_new_attention_order,
                                     num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads , down=True, use_conv=use_conv)

 
        
        resolution*=4
        self.dconv_down3 = ConvGroup(in_channels=base_channels* mult[1],out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_new_attention_order=use_new_attention_order,
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads, down=True, use_conv=use_conv) 

 

        resolution=8
        self.dconv_down4 = ConvGroup(in_channels=base_channels* mult[2],out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                       use_new_attention_order=use_new_attention_order,
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads, down=True, use_conv=use_conv) 
 
        #self.downsample4 = Downsample(channels=base_channels* mult[3])    
        
        resolution=16
        self.bottle_neck = BottleNeck(channels=base_channels* mult[3],attention_resolution=attention_resolution,emb_channels=self.time_emb_dim,dropout=dropout, use_flash_attention=use_flash_attention,
                                      use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  resolution=resolution,  use_new_attention_order=use_new_attention_order,
                                      num_head_channels=num_head_channels, num_heads=num_heads, use_conv=use_conv)  


        #self.upsample4 = Upsample(channels=base_channels* mult[3],use_conv=use_conv) 

        resolution=8
        self.dconv_up4 = ConvGroup(in_channels=base_channels* mult[3] + base_channels* mult[3],out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,up=True, use_conv=use_conv )  
 
        #self.upsample3 = Upsample(channels=base_channels* mult[2],use_conv=use_conv)       
  
        resolution=4
        self.dconv_up3 = ConvGroup(in_channels=base_channels* mult[3] + base_channels* mult[2] ,out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,
                                       num_head_channels=num_head_channels, resolution=resolution , num_heads=num_heads,up=True, use_conv=use_conv)  
 
        #self.upsample2 = Upsample(channels=base_channels* mult[1],use_conv=use_conv)        
      
        resolution=2
        self.dconv_up2 = ConvGroup(in_channels=base_channels* mult[2]  + base_channels* mult[1],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,up=True, use_conv=use_conv )   
 
        #self.upsample1 = Upsample(channels=base_channels* mult[0],use_conv=use_conv)         
 
        resolution=1
        self.dconv_up1 = ConvGroup(in_channels=base_channels* mult[1]  + base_channels* mult[0],out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=self.time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm,  use_flash_attention=use_flash_attention,
                                     use_new_attention_order=use_new_attention_order,
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads,up=True, use_conv=use_conv )   
 
        #self.dconv_up1 =  nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1)  
        #self.conv_last = nn.Conv2d(base_channels* mult[0]//2, img_channels,kernel_size=3,padding=1)  
        self.conv_last = nn.Sequential(
            nn.GroupNorm(groupnorm, base_channels* mult[0]),
            nn.SiLU(),
            zero_module(nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1 ) )
        )
        #self.conv_last = zero_module(nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1) )

  

  def forward(self, x, time ):    
        time_emb = self.time_mlp(time)
        #
        x= self.conv_input(x) 
        conv1 = self.dconv_down1(x,time_emb) 
         
        conv2 = self.dconv_down2(conv1,time_emb) 
        conv3 = self.dconv_down3(conv2,time_emb) 
        conv4 = self.dconv_down4(conv3,time_emb)  

        x = self.bottle_neck(conv4,time_emb)    

        x = self.dconv_up4(torch.cat([x, conv4], dim=1)  ,time_emb)    
        x = self.dconv_up3(torch.cat([x, conv3], dim=1) ,time_emb)  
        x = self.dconv_up2(torch.cat([x, conv2], dim=1) ,time_emb) 
        x = self.dconv_up1(torch.cat([x, conv1], dim=1),time_emb )  
        x = self.conv_last(x)  
        return x


