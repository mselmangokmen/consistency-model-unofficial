
 
import torch  
from torch import nn 
from torchvision.utils import save_image
 
from model.positional_embedding import PositionalEmbedding

from model.residual_convolution_block import ResidualDoubleConv 

import torch.nn.functional as F
class UNet(nn.Module):
    


  def __init__(self,time_emb_dim, img_channels=3,mult=[1,2,4,8],base_channels=64,time_emb_scale=1,group_norm=8):

        super().__init__()   
 
        self.dilation=1 
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        
        self.upsample = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)    
        self.sigmoid = nn.Sigmoid() 
        self.encoder_layers=[]
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )


        self.dconv_down1 = ResidualDoubleConv(img_channels, base_channels* mult[0],group_norm=3) 

        self.dconv_down2 = ResidualDoubleConv(base_channels* mult[0],base_channels* mult[1],group_norm=group_norm) 
        self.dconv_down3 = ResidualDoubleConv(base_channels* mult[1], base_channels* mult[2],group_norm=group_norm)
        self.dconv_down4 = ResidualDoubleConv(base_channels* mult[2], base_channels* mult[3],group_norm=group_norm)

        self.bottle_neck = ResidualDoubleConv(base_channels* mult[3], base_channels* mult[3],group_norm=group_norm)  

        self.dconv_up4 = ResidualDoubleConv(base_channels* mult[3] + base_channels* mult[3], base_channels* mult[3],group_norm=group_norm)
        self.dconv_up3 = ResidualDoubleConv(base_channels* mult[3] + base_channels* mult[2], base_channels* mult[2],group_norm=group_norm)
        self.dconv_up2 = ResidualDoubleConv(base_channels* mult[2] + base_channels* mult[1], base_channels* mult[1],group_norm=group_norm)
        self.dconv_up1 = ResidualDoubleConv(base_channels* mult[1] + base_channels* mult[0], base_channels* mult[0],group_norm=group_norm)
        self.conv_last = ResidualDoubleConv(base_channels* mult[0], img_channels,group_norm=group_norm)  
 
 
          
  def forward(self, x, time ): 
        time_emb = self.time_mlp(time)
        conv1 = self.dconv_down1(x,time_emb) 
        
        x = self.avgpool(conv1)

        # [10,64,32,32] 
        conv2 = self.dconv_down2(x,time_emb)
        # [10,128,32,32] 
        x = self.avgpool(conv2)
        
        # [10,128,16,16] 
        conv3 = self.dconv_down3(x,time_emb)
        # [10,256,16,16] 

        x = self.avgpool(conv3)   

        # [10,256,8,8] 
        conv4 = self.dconv_down4(x,time_emb) 
        x = self.avgpool(conv4)  

        x = self.bottle_neck(x)  

        x = self.upsample(x)     
        
        # [10,1024,8,8]    
        x = torch.cat([x, conv4], dim=1)   
        # x= [10,1024 + 512 ,8,8]    
        x = self.dconv_up4(x,time_emb) 
          # x= [10, 512 ,8,8]    
        x = self.upsample(x)        

          # x= [10, 512 ,16,16]  
        x = torch.cat([x, conv3], dim=1)    
          # x= [10, 512 + 256 ,16,16]       
        x = self.dconv_up3(x,time_emb)

          # x= [10,  256 ,16,16]       
        x = self.upsample(x)        

          # x= [10,  256 ,32,32]    
        x = torch.cat([x, conv2], dim=1)    
          # x= [10,  256 + 128,32,32]      
        x = self.dconv_up2(x,time_emb)

          # x= [10,  128,32,32]     
        x = self.upsample(x)        
          # x= [10,  128,64,64]     
        x = torch.cat([x, conv1], dim=1)   
          # x= [10,  128 + 64,64,64]   
        x = self.dconv_up1(x,time_emb) 
          # x= [10,64,64,64]   
        x = self.conv_last(x)
          # x= [10,1,64,64]   
        out = self.sigmoid(x) 
          # x= [10,1,64,64]   
        return out   

  def loss(self, x, z, t1, t2, ema_model):
        x2 = x + z * t2[:, :, None, None]
        x2 = self(x2, t2)

        with torch.no_grad():
            x1 = x + z * t1[:, :, None, None]
            x1 = ema_model(x1, t1)

        return F.mse_loss(x1, x2)