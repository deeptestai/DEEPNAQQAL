import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

# Define GCN function
def global_contrast_normalization(x: torch.tensor, scale='l1'):
    assert scale in ('l1', 'l2')
    mean = torch.mean(x)
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    elif scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / torch.numel(x)
    x /= x_scale
    return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Calculate GCN min/max values per class
class_min_max = defaultdict(lambda: [float('inf'), float('-inf')])
for image, label in mnist_train:
    gcn_image = global_contrast_normalization(image.squeeze(), scale='l1')
    min_val, max_val = gcn_image.min().item(), gcn_image.max().item()
    class_min_max[label][0] = min(class_min_max[label][0], min_val)
    class_min_max[label][1] = max(class_min_max[label][1], max_val)

# Display the exact min/max values
mnist_gcn_min_max = {label: (vals[0], vals[1]) for label, vals in class_min_max.items()}
print(mnist_gcn_min_max)
