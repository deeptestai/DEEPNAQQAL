import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models

# Custom class to deprocess SVHN-normalized images
#class DeprocessTransform:
 #   def __init__(self, mean, std):
  #      self.mean = torch.tensor(mean).view(3, 1, 1)
   #     self.std = torch.tensor(std).view(3, 1, 1)

   # def __call__(self, image):
        # Ensure correct shape (C, H, W)
    #    if image.shape[-1] == 3 and image.dim() == 3:  # If shape is (32, 32, 3)
     #       image = image.permute(2, 0, 1)  # Convert to (3, 32, 32)
        
        # Deprocess the image (undo SVHN normalization)
      #  image = image * self.std + self.mean
       # image = torch.clamp(image * 255.0, 0, 255)  # Convert back to pixel range [0, 255]
        
       # return image

def vgg16_transfer_svhn(num_classes=2):
    # Load the pretrained VGG16 model
    model_conv = models.vgg16(pretrained=True)
    # Freeze convolutional layers to retain pretrained features
    for param in model_conv.features.parameters():
        param.requires_grad = False

    # Custom classifier with dropout for regularization
    model_conv.classifier = nn.Sequential(
        nn.Linear(25088, 256),
        nn.ReLU(),
       # nn.Dropout(0.5),
        nn.Linear(256, 2))  # Binary classification (ID vs OOD))

    # VGG16-specific normalization values
    vgg16_mean = [0.485, 0.456, 0.406]
    vgg16_std = [0.229, 0.224, 0.225]

    # Data transformations for training and validation
    data_transforms = {
        'train': transforms.Compose([
            #deprocess_transform,  # Undo SVHN normalization
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize for VGG16
            #transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),  # Apply VGG16 normalization
        ]),
        'val': transforms.Compose([
            #deprocess_transform,  
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            #transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
    }

    return model_conv, data_transforms
