import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


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



class ModelTraining():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self):
      self.train_losses = []
      self.test_losses = []
      self.train_acc = []
      self.test_acc = []

    def GetCorrectPredCount(self, pPrediction, pLabels):
      return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

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

        # Calculate loss
        loss = F.nll_loss(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        correct += self.GetCorrectPredCount(pred, target)
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

                correct += self.GetCorrectPredCount(output, target)


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







        