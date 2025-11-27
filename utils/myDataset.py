import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from typing import Any
import numpy as np

class myDataset(Dataset):
    def __init__(self, X, y, transform: transforms.Compose):
        super().__init__()

        if len(X) != len(y):
            raise Exception("X and y are not the same length!")
        
        self.X = X
        self.y = y
        self.transform = transform


        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(np.array(y)), y=np.array(y))



    def __getitem__(self, index: int) -> Any:
        return self.transform(self.X[index]), self.y[index]

    def __len__(self) -> int:
        return len(self.X)
    
    def get_class_weights(self):
        return self.weights
        