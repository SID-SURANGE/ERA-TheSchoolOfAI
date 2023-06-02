# declaring the imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# CODE BLOCK 3 + 4
class Dataset():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])

        # Test data transformations
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    def train_data(self):
        return datasets.MNIST('../data', train=True, download=True, transform=self.train_transforms)

    def test_data(self):
        return datasets.MNIST('../data', train=False, download=True, transform=self.test_transforms)


class Utility():
    """
    This class contains utility functions for our CNN model for MNIST dataset
    1) Verify cuda presence and set device to cuda if available
    2) Plot sample images from the train data
    3) Plot the loss/acc from the training 
    """

    # CODE BLOCK 2
    # self not added in below function to show how we can have diff type of functions in same class
    def set_cuda_if_available():
      cuda = torch.cuda.is_available
      print("CUDA Available?", cuda)

      # if gpu is available, make pytorch to shift the defauly device to gpu for all tensor operations
      device = torch.device("cuda" if cuda else "cpu")
      return device


    # CODE BLOCK 6
    def plot_grid(self, train_loader):

        batch_data, batch_label = next(iter(train_loader)) 
        fig = plt.figure()

        for i in range(12):
          plt.subplot(3,4,i+1)
          plt.tight_layout()
          plt.imshow(batch_data[i].squeeze(0), cmap='gray')
          plt.title(batch_label[i].item())
          plt.xticks([])
          plt.yticks([])

