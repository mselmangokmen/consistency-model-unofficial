
import torch as th
import math
from torch import nn

from architectures.UNET_CT.downsample import Downsample 
POSITIONAL_TYPE='positional'
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def vgg_block( in_channels, out_channels,groupnorm=16):
    layers=[]   
    layers.append(Downsample(channels=in_channels )) 
    layers.append(nn.GroupNorm(groupnorm, out_channels))
    layers.append(nn.SiLU()) 
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) 
    blk = nn.Sequential(*layers)
    
    return blk

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


 