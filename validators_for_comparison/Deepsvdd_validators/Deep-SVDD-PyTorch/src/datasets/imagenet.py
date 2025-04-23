from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
from PIL import Image


class ImgNetDataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class: int = 0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = [normal_class]
        self.outlier_classes = [cls for cls in range(4) if cls != normal_class]

        # Pre-computed min and max values for the "pizza" class
        min_max = {0: (-1224.9273681640625, 2757.67919921875)}  # Adjust for the relevant class

        # ImageNet preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize all images to 224x224
            transforms.ToTensor(),
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l2')),
            transforms.Normalize(
                [min_max[normal_class][0]],
                [min_max[normal_class][1] - min_max[normal_class][0]]
            )
        ])

        # Transform labels: 0 for normal, 1 for outliers
        target_transform = transforms.Lambda(lambda x: 0 if x == normal_class else 1)

        # Load train and test datasets
        train_path = f"{root}/train_imagenet"
        val_path = f"{root}/val_imagenet"
        
        train_set = MyImgNt(root=train_path, transform=transform, target_transform=target_transform)
        val_set = MyImgNt(root=val_path, transform=transform, target_transform=target_transform)

        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx([s[1] for s in train_set.samples], self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        # Use validation set for testing
        self.test_set = val_set


class MyImgNt(ImageFolder):
    """Custom ImageFolder class to include index in the return."""

    def __init__(self, *args, **kwargs):
        super(MyImgNt, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Override the original method to include index in the return.
        Args:
            index (int): Index of the image
        Returns:
            tuple: (image, target, index)
        """
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")  # Convert to RGB
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
