import math
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from model.positional_embedding import PositionalEmbedding

from model.utils import kerras_boundaries


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
print(x.shape)


epoch=10
N = math.ceil(math.sqrt((epoch * (150**2 - 4) / 100) + 4) - 1) + 1
#N = 10
print("N: "+str(N))

boundaries = kerras_boundaries(7.0, 0.002, N, 80.0)
print("boundaries.shape: "+str(boundaries.shape))
print(boundaries)
t = torch.randint(0, N - 1, (x.shape[0], 1))
t_0 = boundaries[t]
t_1 = boundaries[t + 1]

print("shape of t: " + str(t.shape),"shape of t: " + str(t_0.shape),"shape of t: " + str(t_1.shape)) 
 

pos_emb =PositionalEmbedding(dim=128)
pe = pos_emb(t_0)
print("pe.shape "  + str(pe.shape))
print(pe)

freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128  )


args = t_1.float() * freqs[None] 
t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) 

print("t_emb.shape "  + str(t_emb.shape))
print(t_emb)
mu = math.exp(2 * math.log(0.95) / N)

print(mu)