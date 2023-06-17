# ERA-TheSchoolOfAI

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


### **Image Sample**
<hr/>

![Sample](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtvaqtuhUyg9hU2XBm7yhM9LgRYB8xR3Ebzza12nPO43jvIbzncsjhlUBf3LT5EP-PQZo&usqp=CAU)