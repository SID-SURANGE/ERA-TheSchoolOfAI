# import
import torch.nn as nn
from utils import Utility
import torch.nn.functional as F

# create utility class object
utility = Utility()



def get_norm(ntype, channels, groups=4):
   if ntype == 'bn':
      return nn.BatchNorm2d(channels)
   if ntype == 'gn':
      return nn.GroupNorm(num_groups=groups, num_channels=channels)


class Net(nn.Module):
     def __init__(self, normalization='bn'):
       super(Net, self).__init__()
       #Input block
       self.normalize_type = normalization

       self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(16),
            get_norm(self.normalize_type, 24),
            nn.Dropout(0.05)
       ) # output_size = 28

       self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(32),
            get_norm(self.normalize_type, 24),
            nn.Dropout(0.05)
       ) # output_size = 28

       self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=16, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU()
       ) #output size = 28
      
       #TRANSITION BLOCK
       self.pool1 = nn.MaxPool2d((2,2))  #out = 14

       self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3,3),padding=1, bias=False),
            nn.ReLU(),
            get_norm(self.normalize_type, 24),
            nn.Dropout(0.05)
       ) #output = 14

       self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            get_norm(self.normalize_type, 24),
            nn.Dropout(0.05)
       ) # output_size = 14

       self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            get_norm(self.normalize_type, 32),
            nn.Dropout(0.05)
       ) #out = 14

       self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU()
       ) #out = 14

       
       self.pool2 = nn.MaxPool2d((2,2)) #out = 5
       #OUTPUT BLOCK

       self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            get_norm(self.normalize_type, 24),
            nn.Dropout(0.05)
       ) #out = 7

       self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            get_norm(self.normalize_type, 24),
            nn.Dropout(0.05)
       ) #out = 5

       self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            get_norm(self.normalize_type, 32),
            nn.Dropout(0.05)
       ) #out = 3

       self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1

       self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False)
       ) #out = 3

     def forward(self, x):
       x = self.convblock1(x)
       x = self.convblock2(x)
       x = self.convblock3(x)
       x = self.pool1(x)
       x = self.convblock4(x)
       x = self.convblock5(x)
       x = self.convblock6(x)
       x = self.convblock7(x)
       x = self.pool2(x)
       x = self.convblock8(x)
       x = self.convblock9(x)
       x = self.convblock10(x)
       x = self.gap(x)
       x = self.convblock11(x)
       x = x.view(-1, 10)
       return F.log_softmax(x, dim=-1)


# Session 7 Model architecture------------------------------------
class Model7(nn.Module):
     def __init__(self):
       super(Model7, self).__init__()
       #Input block
       self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
       ) # output_size = 28

       #CONV BLOCK 1
       self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05)
       ) # output_size = 26

       self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
       ) #output size = 24
      
       #TRANSITION BLOCK
       self.pool1 = nn.MaxPool2d((2,2))  #out = 12
       self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1,1),padding=0, bias=False),
            nn.ReLU(),
       ) #output = 12

       #CONV BLOCK 2
       self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.05)
       ) # output_size = 10

       self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
       ) #out = 10

       self.pool2 = nn.MaxPool2d((2,2)) #out = 5
       self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
       ) #out = 5

       #OUTPUT BLOCK

       self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
       ) #out = 3

       self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU()
       ) #out = 3

       self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1


     def forward(self, x):
       x = self.convblock1(x)
       x = self.convblock2(x)
       x = self.convblock3(x)
       x = self.pool1(x)
       x = self.convblock4(x)
       x = self.convblock5(x)
       x = self.convblock6(x)
       x = self.pool2(x)
       x = self.convblock7(x)
       x = self.convblock8(x)
       x = self.convblock9(x)
       x = self.gap(x)
       x = x.view(-1, 10)
       return F.log_softmax(x, dim=-1)
     

# Session 6 Model architecture------------------------------------
class Model6(nn.Module):
    def __init__(self):
        super(Model6, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding = 1, bias=False)      # OUT = 28
        self.bn1   = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout(0.05)
        self.conv2 = nn.Conv2d(16, 24, 3, padding = 1, bias=False)     # OUT = 28
        self.bn2   = nn.BatchNorm2d(24)
        self.drop2 = nn.Dropout(0.1)

        self.pool1 = nn.MaxPool2d(2, 2)                  # OUT = 14

        self.conv3 = nn.Conv2d(24, 16, 1, bias=False)     # OUT = 14
        self.bn3   = nn.BatchNorm2d(16)
        self.drop3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(16, 24, 3, padding = 1, bias=False)     # OUT = 14
        self.bn4   = nn.BatchNorm2d(24)
        self.drop4 = nn.Dropout(0.1)
        self.conv41 = nn.Conv2d(24, 32, 3, padding = 1, bias=False)     # OUT = 14
        self.bn41   = nn.BatchNorm2d(32)
        self.drop41 = nn.Dropout(0.1)

        self.pool2 = nn.MaxPool2d(2, 2)                  # OUT = 7

        self.conv5 = nn.Conv2d(32, 24, 1, bias=False)    # OUT = 7
        self.bn5   = nn.BatchNorm2d(24)
        self.drop5 = nn.Dropout(0.1)
        self.conv6 = nn.Conv2d(24, 12, 3, bias=False)    # OUT = 5
        self.bn6   = nn.BatchNorm2d(12)
        self.drop6 = nn.Dropout(0.05)
        self.conv7 = nn.Conv2d(12, 10, 3, bias=False)    # OUT = 3
        
        self.gap1  = nn.AvgPool2d(kernel_size=3)         # OUT = 1

    def forward(self, x):
        x = self.drop2(self.bn2(F.relu(self.conv2(self.drop1(self.bn1(F.relu(self.conv1(x))))))))
        x = self.pool1(x)
        x = self.drop41(self.bn41(F.relu(self.conv41(self.drop4(self.bn4(F.relu(self.conv4(self.drop3(self.bn3(F.relu(self.conv3(x))))))))))))
        x = self.pool2(x)
        x = F.relu(self.conv7(self.bn6(F.relu(self.conv6(self.drop5(self.bn5(F.relu(self.conv5(x)))))))))
        x = self.gap1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
        

# Session 5 Model architecture------------------------------------
class Model5(nn.Module):
    '''
    Define the model architecture
    '''
    #This defines the structure of the NN.
    def __init__(self):
        super(Model5, self).__init__()
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