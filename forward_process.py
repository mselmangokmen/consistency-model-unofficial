



from typing import List
from tqdm import tqdm
import math

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from model.positional_embedding import PositionalEmbedding

from model.unet import ConsistencyModel
from model.utils import kerras_boundaries




def forward_process_for_single_image(x,t,N=1000): 
    
    boundaries = kerras_boundaries(7.0, 0.002, N, 80.0)
    z = torch.randn_like(x) 

    t1 = boundaries[t].view(1, 1, 1) 
    x = x + z * t1 
    return x 


