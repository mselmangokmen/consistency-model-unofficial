

from torch.utils.data import DataLoader

from torchvision.datasets import  CIFAR10
from torchvision import transforms

class Cifar10Loader():

    def __init__(self, batch_size):
        tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

        dataset = CIFAR10(
            "./dataset",
            train=True,
            download=True,
            transform=tf,
        )

        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)


 