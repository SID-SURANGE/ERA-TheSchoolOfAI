# ERA-TheSchoolOfAI
 
The aim of this repository is to define a neural net architecture, which is to be trained with MNIST Dataset. 


### **Usage details**
<hr/>
Below are the details for using the code<br />

- Clone the repository
- Using in Local
    1. Check if system has dedicated graphics card to support GPU training
    2. Verify python environment exist with support for Jupyter notebook, Pytorch
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
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510

- Total parameters - 593,200
- Optimizer - SGD (Stochastic gradient descent)
- Learning rate - 0.01 with LR scheduler
- Batch size - 512 (depends on device where this model gets executed)

