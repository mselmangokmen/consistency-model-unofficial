


from datasetloader import Cifar10Loader
from functions import trainCM_Issolation
import torch
batch_size=64
dataloader = Cifar10Loader(batch_size=batch_size).dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainCM_Issolation(img_channels=3,dataloader=dataloader,n_epochs=100,dbname='Cifar10', device= device)