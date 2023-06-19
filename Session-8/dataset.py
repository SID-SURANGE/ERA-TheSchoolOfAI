

import torch
from torchvision import datasets, transforms

from transforms import Transforms

get_transform = Transforms([],'mnist')

class Dataset():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self, batch_size, dataset_type):
        
        self.batch_size = batch_size
        self.kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

        # MNIST or CIFAR or others
        self.dataset_type = dataset_type

        self.train_transforms, _ = get_transform.Mnist_transforms()

        # Test data transformations
        _ , self.test_transforms = get_transform.Mnist_transforms()


    def train_test_loader(self):

        if self.dataset_type == 'MNIST':
            train = datasets.MNIST('../data', train=True, download=True, transform=self.train_transforms)
            test = datasets.MNIST('../data', train=False, download=True, transform=self.test_transforms)
            
            return torch.utils.data.DataLoader(train, **self.kwargs), torch.utils.data.DataLoader(test, **self.kwargs)
        
        if self.dataset_type == 'CIFAR10':
            train = datasets.CIFAR10('../data', train=True, download=True, transform=self.train_transforms)
            test = datasets.CIFAR10('../data', train=False, download=True, transform=self.test_transforms)

            return torch.utils.data.DataLoader(train, **self.kwargs), torch.utils.data.DataLoader(test, **self.kwargs)