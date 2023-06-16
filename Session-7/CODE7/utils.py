# declaring the imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Utility():
    """
    This class contains utility functions for our CNN model for MNIST dataset
    1) Verify cuda presence and set device to cuda if available
    2) Plot sample images from the train data
    """

    # self not added in below function to show how we can have diff type of functions in same class
    def set_cuda_if_available():
      cuda = torch.cuda.is_available
      print("CUDA Available?", cuda)

      # if gpu is available, make pytorch to shift the defauly device to gpu for all tensor operations
      device = torch.device("cuda" if cuda else "cpu")
      return device


    def plot_grid(self, train_loader):

        batch_data, _ = next(iter(train_loader)) 
        # fig = plt.figure()

        # for i in range(12):
        #   plt.subplot(3,4,i+1)
        #   plt.tight_layout()
        #   plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        #   plt.title(batch_label[i].item())
        #   plt.xticks([])
        #   plt.yticks([])

        figure = plt.figure()
        num_of_images = 60
        for index in range(1, num_of_images + 1):
            plt.subplot(6, 10, index)
            plt.axis('off')
            plt.imshow(batch_data[index].numpy().squeeze(), cmap='gray_r')


    def GetCorrectPredCount(self, pPrediction, pLabels):
      return pPrediction.argmax(dim=1).eq(pLabels).sum().item()