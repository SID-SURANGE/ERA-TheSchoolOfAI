
from torchvision import transforms
import albumentations as A
import cv2

class Transforms():
    """
    This class defines the train/test transforms for our CNN model for MNIST dataset
    """
    def __init__(self) :
        return None


    # def albumentations_transforms(self, train=False, test=False):

    #     if train:
    #         transform = A.Compose([
    #                 A.RandomCrop(width=256, height=256),
    #                 A.HorizontalFlip(p=0.5),
    #                 A.RandomBrightnessContrast(p=0.2),
    #             ])
            
    #     if test:
    #         transform = A.Compose([
    #                 A.RandomCrop(width=256, height=256),
    #                 A.HorizontalFlip(p=0.5),
    #                 A.RandomBrightnessContrast(p=0.2),
    #             ])
            
    #     return transform


    def train_transforms(self):

        #train_transforms = self.albumentations_transforms(train=True)

        train_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-5, 5)),
                transforms.ColorJitter(hue=.02, saturation=.05, brightness=0.005),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        return train_transforms
    

    def test_transforms(self):
        #test_transforms = self.albumentations_transforms(test=True)
        
        test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        return test_transforms
    