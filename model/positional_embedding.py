

import math
import torch 
from torch import nn

class PositionalEmbedding(nn.Module):
    
 
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        if len(t.shape) ==2 : 
            t= torch.squeeze(t,dim=-1)
        emb = torch.outer(t * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb