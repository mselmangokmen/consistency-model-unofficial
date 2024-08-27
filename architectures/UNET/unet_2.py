
 
import torch  
from torch import nn 
from architectures.UNET.bottleneck import BottleNeck
from architectures.UNET.conv_group import ConvGroup
from architectures.UNET.downsample import Downsample
 
from architectures.UNET.positional_embedding import PositionalEmbedding
from architectures.UNET.upsample import Upsample   
 


class UNET_2(nn.Module):
     
  def __init__(self,
        device,  
               img_channels=3,mult=[1,2,4,8],base_channels=64,
        num_res_blocks = 3,
        num_head_channels=-1,
        num_heads=8,
        dropout=0.02,   
        num_classes=10, 
        attention_resolution = [8,16],
        groupnorm=32, 
        use_conv_up=True,
        use_conv_down=True  ):
        
        super().__init__()        
        #self.base_channels=base_channels* 4
        #time_emb_dim = base_channels*mult[-2]
        time_emb_dim = base_channels*4
        self.time_mlp =   PositionalEmbedding(dim=time_emb_dim,device=device ) 
        

        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        self.conv_in = nn.Conv2d(img_channels, base_channels* mult[0],kernel_size=3,padding= 1) 
        #self.conv_in =  nn.Sequential(  nn.Conv2d( img_channels, base_channels* mult[0]//2, 3, padding=1), nn.GroupNorm(groupnorm, base_channels* mult[0]//2),  nn.SiLU()  )
        
        resolution=1
        self.dconv_down1 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[0],
        num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,   
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads  )  
        self.down_sample1=Downsample(channels=base_channels* mult[0], use_conv=use_conv_down)
        
  
        resolution=2
        self.dconv_down2 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,  
                                    
                                     num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads  )
        self.down_sample2=Downsample(channels=base_channels* mult[1], use_conv=use_conv_down) 
        resolution=4
        self.dconv_down3 = ConvGroup(in_channels=base_channels* mult[1],out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,   
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads  ) 
        self.down_sample3=Downsample(channels=base_channels* mult[2], use_conv=use_conv_down)


        resolution=8
        self.dconv_down4 = ConvGroup(in_channels=base_channels* mult[2],out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,  
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,   
                                     num_head_channels=num_head_channels , resolution=resolution, num_heads=num_heads,  ) 
        self.down_sample4=Downsample(channels=base_channels* mult[3], use_conv=use_conv_down)

        resolution=16
        self.bottle_neck =BottleNeck(in_channels=base_channels* mult[3],out_channels=base_channels* mult[3], emb_channels=time_emb_dim,dropout=dropout, 
                                      groupnorm=groupnorm,      
                                      num_head_channels=num_head_channels, num_heads=num_heads,  )  
 
  
        self.up_sample4=Upsample(channels=base_channels* mult[3], use_conv=use_conv_up)
        resolution=8
        self.dconv_up4 = ConvGroup(in_channels=   base_channels* mult[3]+base_channels* mult[3] ,out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,  
                              
                                       num_head_channels=num_head_channels, resolution=resolution , num_heads=num_heads  )  
        

        self.up_sample3=Upsample(channels=base_channels* mult[3], use_conv=use_conv_up)
        resolution=4
        self.dconv_up3 = ConvGroup(in_channels=   base_channels* mult[3] + base_channels* mult[2]  ,out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,  
                              
                                       num_head_channels=num_head_channels, resolution=resolution , num_heads=num_heads  )  
         
         
        self.up_sample2=Upsample(channels=base_channels* mult[2], use_conv=use_conv_up)
        resolution=2
        self.dconv_up2 =  ConvGroup(in_channels=  base_channels* mult[2]+ base_channels* mult[1],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,   
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads   )   
               
 
        self.up_sample1=Upsample(channels=base_channels* mult[1], use_conv=use_conv_up)
        resolution=1
        self.dconv_up1 =  ConvGroup(in_channels= base_channels* mult[1] + base_channels* mult[0],out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,groupnorm=groupnorm,  
                             
                                       num_head_channels=num_head_channels, resolution=resolution, num_heads=num_heads )
        
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
        
         
        x = self.conv_out(x) 
        return x
