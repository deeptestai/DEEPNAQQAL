import torch
import numpy as np
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor

def global_contrast_normalization(x: torch.tensor, scale='l1'):
    """
    Apply global contrast normalization to tensor.
    """
    assert scale in ('l1', 'l2'), "Scale must be 'l1' or 'l2'."

    n_features = int(np.prod(x.shape))

    # Subtract mean
    mean = torch.mean(x)
    x -= mean

    # Apply scaling
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    elif scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    # Normalize
    x /= x_scale

    return x

def compute_classwise_gcn_min_max(dataset):
    """
    Compute class-wise min and max values after applying GCN.

    Args:
        dataset: PyTorch SVHN dataset
    
    Returns:
        dict: Dictionary with min and max values for each class.
    """
    class_min_max = {i: {'min': float('inf'), 'max': float('-inf')} for i in range(10)}

    for img, label in dataset:
        # Convert image to tensor
        img = ToTensor()(img)  # Convert image to tensor with shape [C, H, W]

        # Apply GCN
        gcn_img = global_contrast_normalization(img, scale='l1')

        # Compute min and max for the current class
        current_min = gcn_img.min().item()
        current_max = gcn_img.max().item()

        # Update class-wise min and max
        class_min_max[label]['min'] = min(class_min_max[label]['min'], current_min)
        class_min_max[label]['max'] = max(class_min_max[label]['max'], current_max)

    return class_min_max

# Load the SVHN dataset
svhn_train = SVHN(root='./data', split='train', download=True)

# Compute class-wise GCN min and max
class_min_max = compute_classwise_gcn_min_max(svhn_train)

# Print results
for cls, min_max in class_min_max.items():
    print(f"Class {cls}: Min = {min_max['min']}, Max = {min_max['max']}")
