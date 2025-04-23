import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models

def vgg16_transfer_mnist(num_classes=2):
    # Load the pretrained VGG16 model
    model_conv = models.vgg16(pretrained=True)
    # Freeze convolutional layers to retain pretrained features
    for param in model_conv.features.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer for binary classification (ID vs OOD)
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
            #transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Fix grayscale to RGB
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize for VGG16
            #transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),  # Apply VGG16 normalization
        ]),
        'val': transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Fix grayscale to RGB
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            #transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
    }

    return model_conv, data_transforms
