
from torchvision import transforms
import albumentations as A
import cv2
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, HueSaturationValue, Rotate, RGBShift, Cutout
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class albumentation:
      def __init__(self):
          self.albumentation_transform = Compose([RandomCrop(32,32),
                                                  HorizontalFlip(p=0.5),
                                                  HueSaturationValue(hue_shift_limit=3, sat_shift_limit=2),
                                                  # RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
                                                  Cutout(num_holes=8),
                                                  Normalize(
                                                      mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5],
                                                  ),
                                                  ToTensorV2()
                                                  ])

      def __call__(self, img):
          img = np.array(img)
          img = self.albumentation_transform(image=img)
          return img['image']


class albumentation_test:
      def __init__(self):
          self.albumentation_transform = Compose([
              Normalize(
                  mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5],
              ),
              ToTensorV2()
          ])

      def __call__(self, img):
          img = np.array(img)
          img = self.albumentation_transform(image=img)
          return img['image']
  

# class Transforms():
#     """
#     This class defines the train/test transforms for our CNN model for MNIST dataset
#     """
#     def __init__(self) :
#         return None


#     # def albumentations_transforms(self, train=False, test=False):

#     #     if train:
#     #         transform = A.Compose([
#     #                 A.RandomCrop(width=256, height=256),
#     #                 A.HorizontalFlip(p=0.5),
#     #                 A.RandomBrightnessContrast(p=0.2),
#     #             ])
            
#     #     if test:
#     #         transform = A.Compose([
#     #                 A.RandomCrop(width=256, height=256),
#     #                 A.HorizontalFlip(p=0.5),
#     #                 A.RandomBrightnessContrast(p=0.2),
#     #             ])
            
#     #     return transform


#     def train_transforms(self):

#         #train_transforms = self.albumentations_transforms(train=True)

#         train_transforms = transforms.Compose(
#             [transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation((-5, 5)),
#                 transforms.ColorJitter(hue=.02, saturation=.05, brightness=0.005),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])

#         return train_transforms
    

#     def test_transforms(self):
#         #test_transforms = self.albumentations_transforms(test=True)
        
#         test_transforms = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])

#         return test_transforms
