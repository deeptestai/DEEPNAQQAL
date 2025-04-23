import torch
from models.classifiers.pt_vgg16 import ImageNet_classifier
import torchvision.transforms as transforms

import os

def vgg16_transfer(num_classes=2):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'classifiers', 'imagenet_class.pt')

    model = ImageNet_classifier()
    model_sd = torch.load(filename)
    model.load_state_dict(model_sd)

    for param in model.parameters():
        param.requires_grad = False

    model.reset_last_layer(num_classes)    
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]

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
