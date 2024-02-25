
 
import torch  
from torch import nn 

from architectures.UNET.positional_embedding import PositionalEmbedding
from architectures.UNET.upsample import Upsample
from architectures.UNET.attention_block import AttentionBlock



def zero_module( module):
      """
      Zero out the parameters of a module and return it.
      """
      for p in module.parameters():
          p.detach().zero_()
      return module


