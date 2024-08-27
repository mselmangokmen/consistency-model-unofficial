 
from torchinfo import summary

from architectures.UNET_CT.unet_ct import UNET_CT
from utils.common_functions import get_checkpoint  

model= UNET_CT(  device='cuda:0',img_channels=1, groupnorm=32,
     dropout=0,base_channels=128, num_head_channels=64,
        num_res_blocks=2).to(device=0)
 
 
batch_size = 16
state_dict= get_checkpoint(epoch=20000,model_name='hn_ldct_small')
model.load_state_dict(state_dict)
summary(model, [(batch_size, 1, 32, 32),(batch_size,),(batch_size, 1, 32, 32)])