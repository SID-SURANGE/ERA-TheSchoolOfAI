import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    Define the model architecture
    '''
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  #26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) #24
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) #22
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) #20
        self.fc1 = nn.Linear(4*4*256, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)                   #26
        x = F.relu(F.max_pool2d(self.conv2(x), 2))     #12
        x = F.relu(self.conv3(x), 2)                   #10
        x = F.relu(F.max_pool2d(self.conv4(x), 2))     #4
        x = x.view(-1, 4*4*256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)