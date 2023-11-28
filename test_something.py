import math
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F


def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")

dim=128
scale=1
half_dim =  dim // 2
num_timesteps=500
b=32
t = torch.randint(0, num_timesteps, (b,) )

activation=F.relu
time_bias = nn.Linear(128, 128) 

x= torch.rand(size=(b,128,32,32))
emb = math.log(10000) / half_dim
emb = torch.exp(torch.arange(half_dim ) * -emb)
print(emb.shape)
emb = torch.outer(t *  scale, emb)
emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
out_time= time_bias(activation(emb))[:, :, None, None]
x = x + out_time
print(out_time.shape)


