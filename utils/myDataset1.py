import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from typing import Any
import numpy as np

class myDataset(Dataset):
    def __init__(self, X, y, transform: transforms.Compose, deprocess=False):
        super().__init__()

        if len(X) != len(y):
            raise Exception("X and y are not the same length!")

        self.X = X
        self.y = y
        self.transform = transform
        self.deprocess = deprocess

        # Compute class weights for imbalanced datasets
        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(np.array(y)), y=np.array(y))

        # Original normalization applied during dataset generation
        self.svhn_mean = torch.tensor([0.14354469, 0.14354469, 0.14354469])
        self.svhn_std = torch.tensor([0.29302433, 0.29302433, 0.29302433])

    def __getitem__(self, index: int) -> Any:
        image = self.X[index]
        label = self.y[index]

        # Deprocess the image if required (undo existing normalization)
        if self.deprocess:
            image = self.deprocess_image(image)

        # Apply transformations (resize, normalize for VGG16)
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.X)
    
    def get_class_weights(self):
        return self.weights

    def deprocess_image(self, image):
        """Revert the dataset-specific normalization and scale back to original pixel values."""
        image = image * self.svhn_std.view(3, 1, 1) + self.svhn_mean.view(3, 1, 1)  # Undo normalization
        image = torch.clamp(image * 255.0, 0, 255)  # Scale back to [0,255] range
        return image
