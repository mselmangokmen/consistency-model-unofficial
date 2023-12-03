


from datasetloader import Cifar10Loader
from training_function import trainCM_Issolation

batch_size=64
dataloader = Cifar10Loader(batch_size=batch_size)


trainCM_Issolation(img_channels=1,dataloader=dataloader,n_epochs=100)