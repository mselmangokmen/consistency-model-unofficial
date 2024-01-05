


 
 
import torch

from model.utils import karras_schedule 
  



def forward_process_for_single_image(x,t,N=1000): 
    
    boundaries = karras_schedule(7.0, 0.002, N, 80.0)
    z = torch.randn_like(x) 

    t1 = boundaries[t].view(1, 1, 1) 
    x = x + z * t1 
    return x 


