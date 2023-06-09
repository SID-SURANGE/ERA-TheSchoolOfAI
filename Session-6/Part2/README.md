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
            Conv2d-1           [-1, 16, 28, 28]             144
       BatchNorm2d-2           [-1, 16, 28, 28]              32
           Dropout-3           [-1, 16, 28, 28]               0
            Conv2d-4           [-1, 24, 28, 28]           3,456
       BatchNorm2d-5           [-1, 24, 28, 28]              48
           Dropout-6           [-1, 24, 28, 28]               0
         MaxPool2d-7           [-1, 24, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]             384
       BatchNorm2d-9           [-1, 16, 14, 14]              32
          Dropout-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 24, 14, 14]           3,456
      BatchNorm2d-12           [-1, 24, 14, 14]              48
          Dropout-13           [-1, 24, 14, 14]               0
           Conv2d-14           [-1, 32, 14, 14]           6,912
      BatchNorm2d-15           [-1, 32, 14, 14]              64
          Dropout-16           [-1, 32, 14, 14]               0
        MaxPool2d-17             [-1, 32, 7, 7]               0
           Conv2d-18             [-1, 24, 7, 7]             768
      BatchNorm2d-19             [-1, 24, 7, 7]              48
          Dropout-20             [-1, 24, 7, 7]               0
           Conv2d-21             [-1, 12, 5, 5]           2,592
      BatchNorm2d-22             [-1, 12, 5, 5]              24
           Conv2d-23             [-1, 10, 3, 3]           1,080
        AvgPool2d-24             [-1, 10, 1, 1]               0

        Total params: 19,088
        Trainable params: 19,088
        Non-trainable params: 0

    **Total params: 19,088**


### **Training results**
<hr/>

- Epochs - 15
- Optimizer - SGD (Stochastic gradient descent)
- Learning rate - 0.01 with LR scheduler
- Batch size - 128 (depends on device where this model gets executed)

- Last 4 epochs training - 

        Adjusting learning rate of group 0 to 1.0000e-03.
        Epoch 12
        Train: Loss=0.0587 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:10<00:00, 45.63it/s]
        Test set: Average loss: 0.0163, Accuracy: 9942/10000 (99.42%)

        Adjusting learning rate of group 0 to 1.0000e-03.
        Epoch 13
        Train: Loss=0.0465 Batch_id=468 Accuracy=99.02: 100%|██████████| 469/469 [00:10<00:00, 44.86it/s]
        Test set: Average loss: 0.0159, Accuracy: 9947/10000 (99.47%)

        Adjusting learning rate of group 0 to 1.0000e-03.
        Epoch 14
        Train: Loss=0.0044 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:10<00:00, 45.38it/s]
        Test set: Average loss: 0.0159, Accuracy: 9949/10000 (99.49%)

        Adjusting learning rate of group 0 to 1.0000e-03.
        Epoch 15
        Train: Loss=0.0913 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:10<00:00, 45.42it/s]
        Test set: Average loss: 0.0153, Accuracy: 9952/10000 (99.52%)


- Last epoch results - 
  - **Train accuracy** - 99.11%
  - **Test accuracy** - 99.52%