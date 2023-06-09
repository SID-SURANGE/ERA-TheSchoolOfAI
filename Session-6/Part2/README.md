# Session 6 Part2 - ERA-TheSchoolOfAI
 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-397/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-green.svg)](https://pytorch.org/) [![torchvision 0.15+](https://img.shields.io/badge/torchvision-0.15+-blue.svg)](https://pypi.org/project/torchvision/) [![torch-summary 1.4](https://img.shields.io/badge/torchsummary-1.4+-green.svg)](https://pypi.org/project/torch-summary/)

The repository aims at defining a neural net architecture, which is to be trained with MNIST Dataset using Pytorch Framework. 


### **Usage details**
<hr/>
Below are the details for using the code<br />

- Clone the repository
- Using in Local
    1. Check if system has dedicated graphics card to support GPU training
    2. Verify python > 3.9 environment exist with support for Jupyter notebook, Pytorch
    3. If all exists, open the notebook and run the cells in order
- Using in Colab
    1. Load the .ipynb file into colab, the model.py and utils.py also
    2. Connect the colab to a GPU enabled runtime
    3. Now just run the notebook and play around             


### **Dataset details**
<hr/>

- Dataset Size - Total images - 70000
    1. Train images - 60000
    2. Test images - 10000
- Image profile -
    1. Size - 1x28x28 (Channel x Width x Height)
    2. Grayscale - single channel
- Dataset source - https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html


### **Architecture details**
<hr/>

- Model architecture - 


        Layer (type)               Output Shape         Param #
            Conv2d-1            [-1, 8, 26, 26]              72
       BatchNorm2d-2            [-1, 8, 26, 26]              16
            Conv2d-3           [-1, 16, 24, 24]           1,152
       BatchNorm2d-4           [-1, 16, 24, 24]              32
           Dropout-5           [-1, 16, 24, 24]               0
         MaxPool2d-6           [-1, 16, 12, 12]               0
            Conv2d-7            [-1, 8, 12, 12]             128
       BatchNorm2d-8            [-1, 8, 12, 12]              16
            Conv2d-9           [-1, 16, 10, 10]           1,152
      BatchNorm2d-10           [-1, 16, 10, 10]              32
          Dropout-11           [-1, 16, 10, 10]               0
        MaxPool2d-12             [-1, 16, 5, 5]               0
           Conv2d-13             [-1, 16, 5, 5]             256
      BatchNorm2d-14             [-1, 16, 5, 5]              32
           Conv2d-15             [-1, 10, 3, 3]           1,440
        AvgPool2d-16             [-1, 10, 1, 1]               0

        Total params: 4,328
        Trainable params: 4,328
        Non-trainable params: 0

- Total parameters - 4328


### **Training results**
<hr/>

- Epochs - 15
- Optimizer - SGD (Stochastic gradient descent)
- Learning rate - 0.08 with LR scheduler
- Batch size - 256 (depends on device where this model gets executed)

- Last 4 epochs training - 

        Adjusting learning rate of group 0 to 8.0000e-03.
        Epoch 12
        Train: Loss=0.1224 Batch_id=234 Accuracy=98.54: 100%|██████████| 235/235 [00:13<00:00, 17.21it/s]
        Test set: Average loss: 0.0289, Accuracy: 9912/10000 (99.12%)

        Adjusting learning rate of group 0 to 8.0000e-03.
        Epoch 13
        Train: Loss=0.0277 Batch_id=234 Accuracy=98.49: 100%|██████████| 235/235 [00:13<00:00, 16.92it/s]
        Test set: Average loss: 0.0279, Accuracy: 9914/10000 (99.14%)

        Adjusting learning rate of group 0 to 8.0000e-03.
        Epoch 14
        Train: Loss=0.1069 Batch_id=234 Accuracy=98.56: 100%|██████████| 235/235 [00:13<00:00, 17.00it/s]
        Test set: Average loss: 0.0273, Accuracy: 9916/10000 (99.16%)

        Adjusting learning rate of group 0 to 8.0000e-03.
        Epoch 15
        Train: Loss=0.0166 Batch_id=234 Accuracy=98.59: 100%|██████████| 235/235 [00:13<00:00, 16.92it/s]
        Test set: Average loss: 0.0285, Accuracy: 9914/10000 (99.14%)

        Adjusting learning rate of group 0 to 8.0000e-03.


- Last epoch results - 
  - **Train accuracy** - 98.59%
  - **Test accuracy** - 99.14%