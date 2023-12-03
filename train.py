
from typing import List
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid




def train(dataloader, n_epochs=100, device="cuda:0", img_channels=1) :
    