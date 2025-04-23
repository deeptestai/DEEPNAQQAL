import torch
from models.classifiers.pt_lenet1 import LeNet1
import torchvision.transforms as transforms

import os

def lenet1_transfer(num_classes=2):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname,"classifiers", "lenet1_class.pt")

    model = LeNet1()
    model_sd = torch.load(filename)
    model.load_state_dict(model_sd)

    for param in model.parameters():
        param.requires_grad = False

    in_ftr = model.out.in_features
    model.out = torch.nn.Linear(in_ftr, num_classes)
    
    
    mean = [0.14354469]
    std = [0.29302433]

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
