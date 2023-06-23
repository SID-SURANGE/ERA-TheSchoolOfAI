# imports
import torch
#import tqdm
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


train_losses = []
test_losses = []
train_acc = []
test_acc = []

incorrect_examples = []
incorrect_labels = []
incorrect_pred = []


def train(model, device, train_loader, optimizer):
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

        train_losses.append(loss.cpu().detach().numpy())
        train_acc.append(100*correct/processed)


def test(model, device, test_loader):
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

            # incorrect predictions
            idxs_mask = ((pred == target.view_as(pred))==False).view(-1)
            if idxs_mask.numel(): #if index masks is non-empty append the correspoding data value in incorrect examples
                incorrect_examples.append(data[idxs_mask].squeeze().cpu().numpy())
                incorrect_labels.append(target[idxs_mask].cpu().numpy()) #the corresponding target to the misclassified image
                incorrect_pred.append(pred[idxs_mask].squeeze().cpu().numpy()) 

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))

    return test_loss


# CODE BLOCK 9 Loss/Accuract plot
def plot_loss_accuracy():
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")


def plot_misclassified():
        #fig, axs = plt.subplots(2,2,figsize=(15,10))

        print('Incorrect examples - ', incorrect_examples)
        print('Incorrect labels - ', incorrect_labels)
        print('Incorrect preds - ', incorrect_pred)
