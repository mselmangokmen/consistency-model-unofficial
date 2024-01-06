

import math
import torch 
from torch import nn

class PositionalEmbedding(nn.Module):
    
 
    def __init__(self, dim,device, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale 
        self.device=device
    def forward(self, t): 
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim) * -emb).to(device=self.device) 

        #t= torch.squeeze(t,dim=-1).to(device=self.device)
        #t=torch.squeeze(t,dim=-1).to(device=self.device) 
        emb = torch.outer(t * self.scale, emb).to(device=self.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb