
 
import torch  
from torch import nn  
 
from model.positional_embedding import PositionalEmbedding

from model.residual_convolution_block import ResidualDoubleConv
from model.up_down_sampling import UpDownSampling 

class ConsistencyModel(nn.Module):
    


  def __init__(self,time_emb_dim,device, eps= 0.002, img_channels=3,mult=[1,2,4,8],base_channels=64,time_emb_scale=1,group_norm=8):

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


        self.dconv_down1 = ResidualDoubleConv(img_channels, base_channels* mult[0],group_norm=group_norm,time_emb_dim=time_emb_dim)  
        self.downsample1 = UpDownSampling( channels= base_channels* mult[0],group_norm=group_norm,sampling='down' ,time_emb_dim=time_emb_dim)    

        self.dconv_down2 = ResidualDoubleConv(base_channels* mult[0],base_channels* mult[1],group_norm=group_norm,time_emb_dim=time_emb_dim) 
        self.downsample2 = UpDownSampling( channels= base_channels* mult[1],group_norm=group_norm,sampling='down',time_emb_dim=time_emb_dim )    

        self.dconv_down3 = ResidualDoubleConv(base_channels* mult[1], base_channels* mult[2],group_norm=group_norm,time_emb_dim=time_emb_dim)
        self.downsample3 = UpDownSampling( channels= base_channels* mult[2],group_norm=group_norm,sampling='down',time_emb_dim=time_emb_dim )    

        self.dconv_down4 = ResidualDoubleConv(base_channels* mult[2], base_channels* mult[3],group_norm=group_norm,time_emb_dim=time_emb_dim)
        self.downsample4 = UpDownSampling( channels= base_channels* mult[3],group_norm=group_norm,sampling='down',time_emb_dim=time_emb_dim )    

        self.bottle_neck = ResidualDoubleConv(base_channels* mult[3], base_channels* mult[3],group_norm=group_norm,time_emb_dim=time_emb_dim)  
        self.upsample4 = UpDownSampling( channels= base_channels* mult[3],group_norm=group_norm,sampling='up',time_emb_dim=time_emb_dim )    

        self.dconv_up4 = ResidualDoubleConv(base_channels* mult[3] + base_channels* mult[3], base_channels* mult[3],group_norm=group_norm,time_emb_dim=time_emb_dim)
        self.upsample3 = UpDownSampling( channels= base_channels* mult[3],group_norm=group_norm,sampling='up',time_emb_dim=time_emb_dim )    

        self.dconv_up3 = ResidualDoubleConv(base_channels* mult[3] + base_channels* mult[2], base_channels* mult[2],group_norm=group_norm,time_emb_dim=time_emb_dim)
        self.upsample2 = UpDownSampling( channels= base_channels* mult[2],group_norm=group_norm,sampling='up',time_emb_dim=time_emb_dim )    

        self.dconv_up2 = ResidualDoubleConv(base_channels* mult[2] + base_channels* mult[1], base_channels* mult[1],group_norm=group_norm,time_emb_dim=time_emb_dim)
        self.upsample1 = UpDownSampling( channels= base_channels* mult[1],group_norm=group_norm,sampling='up',time_emb_dim=time_emb_dim )    
        
        self.dconv_up1 = ResidualDoubleConv(base_channels* mult[1] + base_channels* mult[0], base_channels* mult[0],group_norm=group_norm,time_emb_dim=time_emb_dim)
        #self.conv_last = nn.Conv2d(base_channels* mult[0], img_channels,kernel_size=3,padding=1)  
 
        self.conv_last = ResidualDoubleConv(base_channels* mult[0], img_channels,group_norm=3,time_emb_dim=time_emb_dim)  
          
  def forward(self, x, time ): 
        x_original = x.clone()
        time_emb = self.time_mlp(time)
        #print(x.shape)
        conv1 = self.dconv_down1(x,time_emb) 
        
        x = self.downsample1(conv1,time_emb)

        # [10,64,32,32] 
        conv2 = self.dconv_down2(x,time_emb)
        # [10,128,32,32] 
        x = self.downsample2(conv2,time_emb)
        
        # [10,128,16,16] 
        conv3 = self.dconv_down3(x,time_emb)
        # [10,256,16,16] 

        x = self.downsample3(conv3,time_emb)   

        # [10,256,8,8] 
        conv4 = self.dconv_down4(x,time_emb) 
        x = self.downsample4(conv4,time_emb)  

        x = self.bottle_neck(x,time_emb)  

        x = self.upsample4(x,time_emb)     
        
        # [10,1024,8,8]    
        x = torch.cat([x, conv4], dim=1)   
        # x= [10,1024 + 512 ,8,8]    
        x = self.dconv_up4(x,time_emb) 
          # x= [10, 512 ,8,8]    
        x = self.upsample3(x,time_emb)        

          # x= [10, 512 ,16,16]  
        x = torch.cat([x, conv3], dim=1)    
          # x= [10, 512 + 256 ,16,16]       
        x = self.dconv_up3(x,time_emb)

          # x= [10,  256 ,16,16]       
        x = self.upsample2(x,time_emb)        

          # x= [10,  256 ,32,32]    
        x = torch.cat([x, conv2], dim=1)    
          # x= [10,  256 + 128,32,32]      
        x = self.dconv_up2(x,time_emb)

          # x= [10,  128,32,32]     
        x = self.upsample1(x,time_emb)        
          # x= [10,  128,64,64]     
        x = torch.cat([x, conv1], dim=1)   
          # x= [10,  128 + 64,64,64]   
        x = self.dconv_up1(x,time_emb) 
          # x= [10,64,64,64]   
        x = self.conv_last(x,time_emb)
        #x= self.sigmoid
          # x= [10,1,64,64]   
        #out = self.sigmoid(x) 
          # x= [10,1,64,64]   

        time = time - self.eps
        
        # page 26 appendixes 
        c_skip_t = 0.25 / (time.pow(2) + 0.25)
        c_out_t = 0.25 * time / ((time + self.eps).pow(2) + 0.25).pow(0.5)

        return c_skip_t[:, :, None, None] * x_original + c_out_t[:, :, None, None] * x   
