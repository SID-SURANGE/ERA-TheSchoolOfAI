import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import Utility

utility = Utility()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, bias=False)      # OUT = 26
        self.bn1   = nn.BatchNorm2d(8)
        self.drop1 = nn.Dropout(0.05)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)     # OUT = 24
        self.bn2   = nn.BatchNorm2d(16)
        self.drop2 = nn.Dropout(0.15)

        self.pool1 = nn.MaxPool2d(2, 2)                  # OUT = 12

        self.conv3 = nn.Conv2d(16, 8, 1, bias=False)     # OUT = 12
        self.bn3   = nn.BatchNorm2d(8)
        self.drop3 = nn.Dropout(0.05)
        self.conv4 = nn.Conv2d(8, 16, 3, bias=False)     # OUT = 10
        self.bn4   = nn.BatchNorm2d(16)
        self.drop4 = nn.Dropout(0.15)

        self.pool2 = nn.MaxPool2d(2, 2)                  # OUT = 5

        self.conv5 = nn.Conv2d(16, 12, 1, bias=False)    # OUT = 5
        self.bn5   = nn.BatchNorm2d(12)
        self.drop5 = nn.Dropout(0.05)
        self.conv6 = nn.Conv2d(12, 10, 3, padding = 1, bias=False)    # OUT = 5
        self.bn6   = nn.BatchNorm2d(10)
        self.conv7 = nn.Conv2d(10, 10, 3, bias=False)    # OUT = 3
        
        self.gap1  = nn.AvgPool2d(kernel_size=3)         # OUT = 1

    def forward(self, x):
        x = self.drop2(self.bn2(F.relu(self.conv2(self.drop1(self.bn1(F.relu(self.conv1(x))))))))
        x = self.pool1(x)
        x = self.drop4(self.bn4(F.relu(self.conv4(self.drop3(self.bn3(F.relu(self.conv3(x))))))))
        x = self.pool2(x)
        x = F.relu(self.conv7(self.bn6(F.relu(self.conv6(self.drop5(self.bn5(F.relu(self.conv5(x)))))))))
        x = self.gap1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class ModelTraining():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self):
      self.train_losses = []
      self.test_losses = []
      self.train_acc = []
      self.test_acc = []



    def train(self, model, device, train_loader, optimizer):
      model.train()
      pbar = tqdm(train_loader)

      train_loss = 0
      correct = 0
      processed = 0

      for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)
        #print(f'model pred shape - {pred.shape}')

        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += utility.GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

      self.train_acc.append(100*correct/processed)
      self.train_losses.append(train_loss/len(train_loader))


    def test(self, model, device, test_loader):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                correct += utility.GetCorrectPredCount(output, target)


        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    # CODE BLOCK 9 Loss/Accuract plot
    def plot_loss_accuracy(self):
          fig, axs = plt.subplots(2,2,figsize=(15,10))
          axs[0, 0].plot(self.train_losses)
          axs[0, 0].set_title("Training Loss")
          axs[1, 0].plot(self.train_acc)
          axs[1, 0].set_title("Training Accuracy")
          axs[0, 1].plot(self.test_losses)
          axs[0, 1].set_title("Test Loss")
          axs[1, 1].plot(self.test_acc)
          axs[1, 1].set_title("Test Accuracy")







        