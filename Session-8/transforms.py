
from torchvision import transforms


class Transforms():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self, tranforms_list, dataset):
        self.transform_list = tranforms_list
        self.dataset = dataset

    def Mnist_transforms():

        train_transforms = transforms.Compose([
                transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                transforms.Resize((28, 28)),
                transforms.RandomRotation((-15., 15.), fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])

        test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])

        return train_transforms, test_transforms
    

    def CIFAR_tranforms():
        return ''
    
    def custom_transfroms():
        return ''