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







        