import torch
import torch.nn as nn
from models.classifiers.pt_svhn_classifier import SVHN_classifier
import torchvision.transforms as transforms

import os

def svhn_transfer(num_classes=2):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname,"classifiers", "svhn_class.pt")
    # Check if file exists before loading
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found at: {filename}")
    model = SVHN_classifier()
    model_sd = torch.load(filename)
    model.load_state_dict(model_sd)

    # Freeze all layers except the last layer for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    model.reset_last_layer(num_classes)    
    # Unfreeze only the last layer for training
    for param in model.conv7.parameters():
        param.requires_grad = True  
    #mean = [0.4148]
    #std = [0.2801]
    mean = [0.14354469, 0.14354469, 0.14354469]
    std = [0.29302433, 0.29302433, 0.29302433]
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize(size=(224,224)),
            #transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean, std),
            #transforms.Grayscale(num_output_channels=3)
        ]),
        'val': transforms.Compose([
            #transforms.Resize(size=(224,224)),
            #transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean, std),
            #transforms.Grayscale(num_output_channels=3)
        ]),
    }

    return model, data_transforms
