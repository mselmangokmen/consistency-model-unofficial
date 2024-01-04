
 
import torch  
from torch import nn  
from torch.nn import functional as F
from model.bottleneck import BottleNeck
from model.conv_group import ConvGroup
from model.downsample import Downsample

from model.positional_embedding import PositionalEmbedding
from model.upsample import Upsample 



class UNET(nn.Module):
    


  def __init__(self,time_emb_dim,device, eps= 0.002, 
               img_channels=3,mult=[1,2,4,8],base_channels=64,
               time_emb_scale=1,
        num_res_blocks = 3,
        dropout=0.02, 
        attention_resolution = 8,
        groupnorm=16,
        use_scale_shift_norm=False):

        super().__init__()   
        self.eps = eps
        self.dilation=1 
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        
        #self.upsample = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)    
        
        self.sigmoid = nn.Sigmoid() 
        self.encoder_layers=[]
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(dim=base_channels, scale=time_emb_scale,device=device),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )


        self.conv_input = nn.Conv2d(3, base_channels//2,kernel_size=3,padding=1)  
        self.dconv_down1 = ConvGroup(in_channels=base_channels//2,out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)  
        self.downsample1 = Downsample(channels=base_channels* mult[0])    
        
        self.dconv_down2 = ConvGroup(in_channels=base_channels* mult[0],out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)  
        self.downsample2 = Downsample(channels=base_channels* mult[1])    
        

        self.dconv_down3 = ConvGroup(in_channels=base_channels* mult[1],out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)  
        self.downsample3 = Downsample(channels=base_channels* mult[2])    

   
        self.dconv_down4 = ConvGroup(in_channels=base_channels* mult[2],out_channels=base_channels* mult[3],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)  
        self.downsample4 = Downsample(channels=base_channels* mult[3])    


        self.bottle_neck = BottleNeck(channels=base_channels* mult[3],attention_resolution=attention_resolution,emb_channels=time_emb_dim,dropout=dropout,
                                      use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)  


        self.upsample4 = Upsample(channels=base_channels* mult[3])     
        self.dconv_up4 = ConvGroup(in_channels=base_channels* mult[3] + base_channels* mult[3],out_channels=base_channels* mult[2],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)  
         

        self.upsample3 = Upsample(channels=base_channels* mult[2])       
  
        self.dconv_up3 = ConvGroup(in_channels=base_channels* mult[2] + base_channels* mult[2] ,out_channels=base_channels* mult[1],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)  
         
        self.upsample2 = Upsample(channels=base_channels* mult[1])        
      
        self.dconv_up2 = ConvGroup(in_channels=base_channels* mult[1]  + base_channels* mult[1],out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)   
        
        self.upsample1 = Upsample(channels=base_channels* mult[0])         
 
        self.dconv_up1 = ConvGroup(in_channels=base_channels* mult[0]  + base_channels* mult[0],out_channels=base_channels* mult[0],num_res_blocks=num_res_blocks,attention_resolution=attention_resolution,
                                     emb_channels=time_emb_dim,dropout=dropout,use_scale_shift_norm=use_scale_shift_norm,groupnorm=groupnorm)   
        
        #self.dconv_up1 =  nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1)  
        self.conv_last = nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1)  
 
        #self.conv_last = nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1)  
          
  def forward(self, x, time ): 
        x_original = x.clone()
        time_emb = self.time_mlp(time)
        #
        x= self.conv_input(x)
        #print(x.shape)
        conv1 = self.dconv_down1(x,time_emb) 
        
        x = self.downsample1(conv1)

        # [10,64,32,32] 
        conv2 = self.dconv_down2(x,time_emb)
        # [10,128,32,32] 
        x = self.downsample2(conv2)
        
        # [10,128,16,16] 
        conv3 = self.dconv_down3(x,time_emb)
        # [10,256,16,16] 

        x = self.downsample3(conv3)   

        # [10,256,8,8] 
        conv4 = self.dconv_down4(x,time_emb) 

        x = self.downsample4(conv4)  

        x = self.bottle_neck(x,time_emb)  

        x = self.upsample4(x)     
          
 

        x = self.dconv_up4(torch.cat([x, conv4], dim=1)  ,time_emb)  

        x = self.upsample3(x)        

 
        #x = torch.cat([x, att3], dim=1)    
          # x= [10, 512 + 256 ,16,16]       
        x = self.dconv_up3(torch.cat([x, conv3], dim=1) ,time_emb)

          # x= [10,  256 ,16,16]       
        x = self.upsample2(x)        

          # x= [10,  256 ,32,32]    
         
        #x = torch.cat([x, att2], dim=1)    
          # x= [10,  256 + 128,32,32]      
        x = self.dconv_up2(torch.cat([x, conv2], dim=1) ,time_emb)

          # x= [10,  128,32,32]     
        x = self.upsample1(x)        
          # x= [10,  128,64,64]     
         
        #x = torch.cat([x, att1], dim=1)   
          # x= [10,  128 + 64,64,64]   
        x = self.dconv_up1(torch.cat([x, conv1], dim=1),time_emb ) 
          # x= [10,64,64,64]   
        x = self.conv_last(x) 

        time = time - self.eps
        
        # page 26 appendixes 
        c_skip_t = 0.25 / (time.pow(2) + 0.25)
        c_out_t = 0.25 * time / ((time + self.eps).pow(2) + 0.25).pow(0.5)

        return c_skip_t[:, :, None, None] * x_original + c_out_t[:, :, None, None] * x   
