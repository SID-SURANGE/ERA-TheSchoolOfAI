import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils import Utility

utility = Utility()


class Net(nn.Module):
     def __init__(self):
       super(Net, self).__init__()
       #Input block
       self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
       ) # output_size = 28

       #CONV BLOCK 1
       self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
       ) # output_size = 26

       self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU()  
       ) #output size = 24
      
       #TRANSITION BLOCK
       self.pool1 = nn.MaxPool2d((2,2))  #out = 12
       self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1,1),padding=0, bias=False),
            nn.ReLU()
       ) #output = 12

       #CONV BLOCK 2
       self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
       ) # output_size = 10

       self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU()  
       ) #out = 8
       
       self.pool2 = nn.MaxPool2d((2,2))  #out = 4

    
       #OUTPUT BLOCK

       self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU()  
       ) #out = 4

       self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(4,4), padding=0, bias=False),
            #nn.ReLU()  
       ) #out = 1


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
       x = x.view(-1, 10)
       return F.log_softmax(x, dim=-1)


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
        pbar = tqdm(train_loader) #, position=0, leave=True)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

            self.train_losses.append(loss.cpu().detach().numpy())
            self.train_acc.append(100*correct/processed)


    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_losses.append(test_loss)
        self.test_acc.append(100. * correct / len(test_loader.dataset))

  


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







        