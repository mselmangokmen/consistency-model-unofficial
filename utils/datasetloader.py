

from torch.utils.data import DataLoader
from PIL import Image 
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import  CIFAR10, CelebA, ImageNet, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import ImageFolder
from PIL import Image
import random

import numpy as np 
import os 
 
import skimage.transform as st


class SkecthTrainDataset(Dataset):
    def __init__(self, transform=None):
        
        photo_path_train= 'dataset/sketch_dataset/train/photos'
        sketch_path_train= 'dataset/sketch_dataset/train/sketches' 
        photo_img_names =os.listdir(photo_path_train)
        self.photo_paths = [os.path.join(photo_path_train, file) for file in photo_img_names if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.sketch_paths = [os.path.join(sketch_path_train, file) for file in photo_img_names if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.photo_paths)

    def __getitem__(self, idx):
        
        photo_path = self.photo_paths[idx]
        sketch_path= self.sketch_paths[idx]
        photo_img = Image.open(photo_path).convert('RGB').resize(size=(128,128))
        sketch_img = Image.open(sketch_path).convert('RGB').resize(size=(128,128))
        if self.transform:
            photo_img = self.transform(photo_img)
            sketch_img = self.transform(sketch_img)
        return photo_img, sketch_img, idx


class SkecthTestDataset(Dataset):
    def __init__(self, transform=None):
        
        photo_path_train= 'dataset/sketch_dataset/test/photos'
        sketch_path_train= 'dataset/sketch_dataset/test/sketches' 
        photo_img_names =os.listdir(photo_path_train)
        self.photo_paths = [os.path.join(photo_path_train, file) for file in photo_img_names if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.sketch_paths = [os.path.join(sketch_path_train, file) for file in photo_img_names if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.photo_paths)

    def __getitem__(self, idx):
        
        photo_path = self.photo_paths[idx]
        sketch_path= self.sketch_paths[idx]
        photo_img = Image.open(photo_path).convert('RGB').resize(size=(128,128))
        sketch_img = Image.open(sketch_path).convert('RGB').resize(size=(128,128))
        if self.transform:
            photo_img = self.transform(photo_img)
            sketch_img = self.transform(sketch_img)
        return photo_img, sketch_img, idx

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


 
class MNISTLoader(): 
    def __init__(self, batch_size,shuffle=False,rank=0):
        tf = transforms.Compose(   [
            #transforms.RandomHorizontalFlip(),   
            transforms.Resize((28, 28)),
              transforms.ToTensor(),  
               #transforms.Lambda(lambda x: (x * 2) - 1)
            transforms.Normalize((0.5,), (0.5) ),
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
 
        self.dataset = MNIST(root='dataset', download=True, transform=tf )
        
        #dataset = CustomImageDataset(folder_path='dataset/img_align_celeba', transform=tf)    
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=shuffle,
                                      sampler=DistributedSampler(self.dataset, seed=42, rank=rank,shuffle=shuffle, drop_last=True) )


 
class CelebALoader(): 
    def __init__(self, batch_size,shuffle=False):
        tf = transforms.Compose(   [
            transforms.RandomHorizontalFlip(),   
            transforms.Resize((64, 64)),
              transforms.ToTensor(),  
               #transforms.Lambda(lambda x: (x * 2) - 1)
            transforms.Normalize((0.5,), (0.5)),
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
 
        dataset = CelebA(root='dataset', download=True, transform=tf )

        #dataset = CustomImageDataset(folder_path='dataset/img_align_celeba', transform=tf)    
        self.dataloader = DataLoader(dataset, batch_size=batch_size,pin_memory=True, shuffle=shuffle,sampler=DistributedSampler(dataset, shuffle=True))


class CelebAFIDLoader(): 
    def __init__(self, batch_size,shuffle=True):
        tf = transforms.Compose(   [
            transforms.RandomHorizontalFlip(),   
            transforms.Resize((64, 64)),
              transforms.ToTensor(),   
        ]
    )
        #output[channel] = (input[channel] - mean[channel]) / std[channel]
 
        dataset = CelebA(root='dataset', download=True, transform=tf )

        #dataset = CustomImageDataset(folder_path='dataset/img_align_celeba', transform=tf)    
        self.fid_loader = DataLoader(dataset, batch_size=batch_size,pin_memory=True, shuffle=shuffle)


 
  



class ImageNetLoader():

    def __init__(self, batch_size,shuffle=False,rank=0):
        tf = transforms.Compose(   [
            transforms.RandomHorizontalFlip(),    
            transforms.Resize((64, 64)), 
              transforms.ToTensor(),  
               #transforms.Lambda(lambda x: (x * 2) - 1)
            transforms.Normalize((0.5,), (0.5) ),
        ]
    )  
         
        self.dataset = ImageFolder('dataset/imagenet', transform=tf)
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        
        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])
        
        # Create Distributed Samplers for training and testing sets
        self.train_sampler = DistributedSampler(self.train_dataset, seed=42, rank=rank, drop_last=True) 
        
        # Create DataLoaders for training and testing sets
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=self.train_sampler
        )
         
class ImageNetFidLoader():

    def __init__(self, batch_size,shuffle=False,rank=0):
        tf = transforms.Compose(   [
            transforms.RandomHorizontalFlip(),    
            transforms.Resize((64, 64)), 
              transforms.ToTensor(),   
        ]
    )  
         
        self.dataset = ImageFolder('dataset/imagenet', transform=tf)
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
         
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])
        
        # Create Distributed Samplers for training and testing sets
        self.test_sampler = DistributedSampler(self.test_dataset, seed=42, rank=rank, drop_last=True) 
         
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False, 
            sampler=self.test_sampler
        )

class Cifar10LoaderNotParallel():

    def __init__(self, batch_size,shuffle=True ):
         
        tf = transforms.Compose(   [ 
            transforms.RandomHorizontalFlip(),   
              transforms.ToTensor(),   
            transforms.Normalize((0.5,), (0.5) ),
        ]
    ) 
        self.dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
        )
        

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=shuffle )


class Cifar10LoaderVAE():

    def __init__(self, batch_size,shuffle=False,rank=0):
         
        tf = transforms.Compose(   [ 
            transforms.RandomHorizontalFlip(),   
              transforms.ToTensor()
        ]
    ) 
        self.dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
        )
        

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=False,
                                     sampler=DistributedSampler(self.dataset, seed=42, rank=rank,shuffle=shuffle, drop_last=True))





class Cifar10FIDLoader():

    def __init__(self, batch_size ):
         
        tf = transforms.Compose(   [  
              transforms.ToTensor(),     
        ]
    ) 
        self.dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
            
        )
        
        self.fid_loader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=True )



class Cifar10Loader():

    def __init__(self, batch_size,shuffle=False,rank=0):
        
        mean = (0.49139968, 0.48215827 ,0.44653124)
        std = (0.24703233,  0.24348505,  0.26158768)
        
        tf = transforms.Compose(   [ 
            transforms.RandomHorizontalFlip(),   
              transforms.ToTensor(),    
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5) )
            #transforms.Normalize((0.5,), (0.5)   ),
            #transforms.Normalize(mean, std)
        ]
    ) 
        self.dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
            
        )
        
        self.test_set = CIFAR10(
            "./dataset",
            train=False,
            download=True,
            transform=tf, 
        ) 
        self.train_dataloader = DataLoader(self.dataset, batch_size=batch_size,pin_memory=True, shuffle=False, 
                                     sampler=DistributedSampler(self.dataset, seed=42, rank=rank,shuffle=shuffle, drop_last=True))


        self.test_dataloader = DataLoader(self.test_set ,batch_size=batch_size,pin_memory=True, shuffle=True )



class ButterflyDatasetLoader:
    def __init__(self, batch_size, image_size,rank):   

        tf = transforms.Compose([

            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5) , (0.5,) ),
        ])  
        self.dataset = ImageFolder('dataset/butterflies256', transform=tf)
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        
        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])
        
        # Create Distributed Samplers for training and testing sets
        self.train_sampler = DistributedSampler(self.train_dataset, seed=42, rank=rank, drop_last=True) 
        
        # Create DataLoaders for training and testing sets
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=self.train_sampler
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True
        )



class VOCDatasetLoader:
    def __init__(self, batch_size,image_size,rank):   

        tf = transforms.Compose([

            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5) , (0.5,) ),
        ])  
        self.dataset = ImageFolder('dataset/VOC2012', transform=tf)
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        
        # Split the dataset into training and testing sets
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])
        
        # Create Distributed Samplers for training and testing sets
        self.train_sampler = DistributedSampler(self.train_dataset, seed=42, rank=rank, drop_last=True, shuffle=True) 
        
        # Create DataLoaders for training and testing sets
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=self.train_sampler
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True
        )



 
class MyDatasetLDCT(Dataset):
    def __init__(self, x, y, max_val,min_val , aug):
        self.x_images = x  
        self.y_images = y 
        self.min_val=min_val
        self.max_val=max_val
        self.aug=aug
    def __len__(self):
        return len(self.x_images)

    def __getitem__(self, idx):
        x_image = self.x_images[idx].astype(np.float32)
        y_image = self.y_images[idx].astype(np.float32)
 

        x_image = st.rescale(x_image, 1/2   )

        y_image = st.rescale(y_image, 1/2   ) 

        x_image=np.expand_dims(x_image, axis=0)
        y_image=np.expand_dims(y_image, axis=0)  
        if self.aug and random.random() > 0.5:
            x_image = np.fliplr(x_image)
            y_image = np.fliplr(y_image) 
        
        if self.aug and random.random() > 0.5:
            x_image = np.flipud(x_image)
            y_image = np.flipud(y_image) 
        
        if self.aug and random.random() > 0.5:
            rotations = random.choice([1, 2, 3])  # 90, 180 veya 270 derece
            x_image = np.rot90(x_image, rotations, axes=(1, 2))
            y_image = np.rot90(y_image, rotations, axes=(1, 2))
        #DGX 
        #if self.aug and random.random() > 0.5:
        #    rotations = random.choice([1, 2, 3])  # 90, 180 veya 270 derece
        #    x_image = np.rot90(x_image, rotations, axes=(1, 2))
        #    y_image = np.rot90(y_image, rotations, axes=(1, 2))
        #print('x_image amax: ', np.amax(x_image))
        #print('x_image amin: ', np.amin(x_image)) 
        
        x_image = x_image * 2 - 1
        y_image = y_image * 2 - 1
        #y_image=np.clip(y_image)
        #x_image=np.clip(x_image)
        return x_image, y_image
    


class LDCTDatasetLoader:
    def __init__(self, batch_size, rank=0):
        #dataset_path = "dataset/"
        dataset_path = "dataset/LDCT_npy"

        # Eğitim ve doğrulama verilerini yükle
        x_train = np.load(os.path.join(dataset_path, 'x_train.npy'))
        y_train = np.load(os.path.join(dataset_path, 'y_train.npy'))
        x_val = np.load(os.path.join(dataset_path, 'x_val.npy'))
        y_val = np.load(os.path.join(dataset_path, 'y_val.npy'))


        self.max_val = np.amax(x_train)
        self.min_val = np.amin(x_train)
        train_set = MyDatasetLDCT(x=x_train, y=y_train, max_val=self.max_val, min_val=self.min_val, aug=True)
        val_set = MyDatasetLDCT(x=x_val, y=y_val, max_val=self.max_val, min_val=self.min_val, aug=False)
 
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True,
            sampler=DistributedSampler(train_set, seed=42,rank=rank,drop_last=True)),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True,
            sampler=DistributedSampler(val_set, seed=42,rank=rank,drop_last=True))
        }
        self.dataloaders = dataloaders

    def getDataLoader(self):
        return self.dataloaders, self.max_val,self.min_val


class ButterflyDatasetLoaderNotParallel:
    def __init__(self, batch_size, image_size):   

        tf = transforms.Compose([

            transforms.Resize(image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])  
        self.dataset = ImageFolder('dataset/butterflies256', transform=tf)
        self.dataloader= DataLoader(
            self.dataset,
            batch_size=batch_size,  
            pin_memory=True 
        )

 
   
  
class CelebALoader128:
    def __init__(self, batch_size):
        # Transformations: Convert to tensor, normalize, and resize
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5) ),
        ]) 
        
        dataset = CustomImageDataset(folder_path='dataset/celeba/img_align_celeba_resized', transform=tf)  
        self.dataloader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=4)




class SketchTestsetLoader:
    def __init__(self, batch_size): 
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5) ),
        ])  
        dataset = SkecthTestDataset( transform=tf)  
        self.dataloader = DataLoader(dataset=dataset,batch_size=batch_size,  shuffle=True,num_workers=4)


 

class SketchDatasetLoader:
    def __init__(self, batch_size): 
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5) ),
        ])  
        dataset = SkecthTrainDataset( transform=tf)  
        self.dataloader = DataLoader(dataset=dataset,batch_size=batch_size,  shuffle=False,sampler=DistributedSampler(dataset))

class MyDatasetSOCO(Dataset):
    def __init__(self, data):
        self.data=data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        real_img = self.data[idx,0,:,:].astype(np.float32)
        cr_img = self.data[idx,1,:,:].astype(np.float32)
        obl_img = self.data[idx,2,:,:].astype(np.float32)
        zcut_img = self.data[idx,3,:,:].astype(np.float32) 
        tf = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x/255) , 
            transforms.Lambda(lambda x: (x * 2) - 1)
        ]) 
        real_img=tf(real_img)
        cr_img=tf(cr_img)
        obl_img=tf(obl_img)
        zcut_img=tf(zcut_img)
        return real_img, cr_img,obl_img,zcut_img
    

class MyDatasetSOCOSINGLE(Dataset):
    def __init__(self, data):
        self.data=data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        real_img = self.data[idx,0,:,:].astype(np.float32)
        pert_img = self.data[idx,1,:,:].astype(np.float32) 
        tf = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x/255) , 
            transforms.Lambda(lambda x: (x * 2) - 1)
        ]) 
        real_img=tf(real_img)
        pert_img=tf(pert_img) 
        return real_img, pert_img 
    

class SOCODatasetLoader:
    def __init__(self, batch_size, rank=0, shuffle=True ): 
 
        train = np.load('dataset/SOCO_train.npy') 
        test = np.load('dataset/SOCO_test.npy')  

 
        train_set = MyDatasetSOCO(data=train )
        test_set = MyDatasetSOCO(data=test )
 
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                 sampler=DistributedSampler(train_set, seed=42, rank=rank,shuffle=shuffle, drop_last=True) ),
            'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True , 
                                sampler=DistributedSampler(test_set, seed=42, rank=rank,shuffle=shuffle, drop_last=True))
        }
        self.dataloaders = dataloaders

    def getDataLoader(self):
        return self.dataloaders 

    

class SOCOSINGLEDatasetLoader:
    def __init__(self, batch_size, rank=0, shuffle=True ): 
 
        train = np.load('dataset/SOCO_train_single.npy') 
        test = np.load('dataset/SOCO_test_single.npy')  

 
        train_set = MyDatasetSOCOSINGLE(data=train )
        test_set = MyDatasetSOCOSINGLE(data=test )
 
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                 sampler=DistributedSampler(train_set, seed=42, rank=rank,shuffle=shuffle, drop_last=True) ),
            'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True , 
                                sampler=DistributedSampler(test_set, seed=42, rank=rank,shuffle=shuffle, drop_last=True))
        }
        self.dataloaders = dataloaders

    def getDataLoader(self):
        return self.dataloaders 