

from torch.utils.data import DataLoader
from PIL import Image 
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import  CIFAR10, CelebA
from torchvision import transforms
import os
 

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure 3 color channels
        if self.transform:
            image = self.transform(image)
        return image, idx


class Cifar10Loader():

    def __init__(self, batch_size,shuffle=False):
        tf = transforms.Compose(   [  transforms.ToTensor(),   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
        self.dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
        )

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=shuffle,sampler=DistributedSampler(self.dataset))

 

class Cifar10LoaderMPI():

    def __init__(self, batch_size,shuffle=False):
        tf = transforms.Compose(   [  transforms.ToTensor(),   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
        self.dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
        )

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=shuffle,sampler=DistributedSampler(self.dataset))

 
class Cifar10LoaderNotParallel():

    def __init__(self, batch_size,shuffle=False):
        tf = transforms.Compose(   [  transforms.ToTensor(),   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
        self.dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
        )

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=shuffle)

 
class Cifar10LoaderNoNormalization():

    def __init__(self, batch_size):
        tf = transforms.Compose(   [  transforms.ToTensor()
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
        dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
        )

        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)

 
class CelebALoader(): 
    def __init__(self, batch_size):
        tf = transforms.Compose(   [  transforms.ToTensor(),   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Resize((128,128))
                                    
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
        dataset = CelebA(
            "./dataset", 
            download=True,
            transform=tf,
        )

        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)


class CelebALoader128:
    def __init__(self, batch_size):
        # Transformations: Convert to tensor, normalize, and resize
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ]) 
        
        dataset = CustomImageDataset(folder_path='dataset/celeba/img_align_celeba_resized', transform=tf)  
        self.dataloader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=4)


 
 