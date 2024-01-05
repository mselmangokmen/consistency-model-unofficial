


from datasetloader import CelebALoader128
from functions import trainCM_Issolation
import torch
import gc
from torch import nn

from model.unet import UNET 
 
batch_size=64
dataloader = CelebALoader128(batch_size=batch_size).dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
img_channels=3
time_emb_dim=64
base_channels=32
num_res_blocks=1
n_epochs=100
dbname='CelabA_unet'
lr=1e-5
hideProgressBar=True 
torch.cuda.empty_cache()
gc.collect()
model = UNET( img_channels=img_channels,  device=device,time_emb_dim=time_emb_dim,base_channels=base_channels,num_res_blocks=num_res_blocks)
ema_model = UNET( img_channels=img_channels,  device=device,time_emb_dim=time_emb_dim,base_channels=base_channels,num_res_blocks=num_res_blocks)
model= nn.DataParallel(model).to(device=device)
ema_model= nn.DataParallel(ema_model).to(device=device)
trainCM_Issolation(model=model,ema_model=ema_model,dataloader=dataloader,n_epochs=n_epochs,dbname=dbname,  lr=lr, device= device,hideProgressBar=False)
