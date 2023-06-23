

import torch
from torchvision import datasets, transforms

from transforms import Transforms

get_transform = Transforms()


class Dataset():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self, batch_size):
        
        self.batch_size = batch_size
        self.kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

        # train data tranforms
        self.train_transforms = get_transform.train_transforms()

        # Test data transformations
        self.test_transforms = get_transform.test_transforms()


    def train_loader(self):
        """
        Train loader for the dataset
        """

        train = datasets.CIFAR10('../data', train=True, download=True, transform=self.train_transforms)
        
        return torch.utils.data.DataLoader(train, **self.kwargs)
        

    def test_loader(self):
        """
        Test loader for the dataset
        """

        test = datasets.CIFAR10('../data', train=False, download=True, transform=self.test_transforms)
        
        return torch.utils.data.DataLoader(test, **self.kwargs)