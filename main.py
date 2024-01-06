


from datasetloader import CelebALoader128, Cifar10Loader
from functions import trainCM_Issolation, trainCM_Issolation
import torch
import gc
from torch import nn

from model.unet import UNET 
 
batch_size=128
dataloader = Cifar10Loader(batch_size=batch_size).dataloader
device = torch.device("cuda") 

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
img_channels=3
time_emb_dim=128
base_channels=128
num_res_blocks=3
model_name='cifar10_unet'
lr=1e-5
hideProgressBar=True 
training_mult= 1000
total_training_steps= 800  * training_mult
torch.cuda.empty_cache()
gc.collect()
model = UNET( img_channels=img_channels,  device=device,time_emb_dim=time_emb_dim,base_channels=base_channels,num_res_blocks=num_res_blocks).to(device=device)
ema_model = UNET( img_channels=img_channels,  device=device,time_emb_dim=time_emb_dim,base_channels=base_channels,num_res_blocks=num_res_blocks).to(device=device)
#model=model.half()
#ema_model=ema_model.half()
#model.convert_to_fp16()
#ema_model.convert_to_fp16()
student_model= nn.DataParallel(model).to(device=device)

teacher_model= nn.DataParallel(ema_model).to(device=device)

#trainCM_Issolation(model=model,ema_model=ema_model,dataloader=dataloader,dbname=dbname,  lr=lr, device= device,hideProgressBar=False)
trainCM_Issolation(dataloader=dataloader,student_model=student_model,total_training_steps=total_training_steps,teacher_model=teacher_model,model_name=model_name,device=device)
