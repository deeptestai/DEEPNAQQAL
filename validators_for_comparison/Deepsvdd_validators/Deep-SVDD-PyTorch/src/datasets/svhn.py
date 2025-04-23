from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import SVHN
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms


class SVHN_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class (use dummy for now)
        min_max = [
                (-9.8109712600708, 11.722309112548828),
                (-11.558295249938965, 11.18669605255127),
                (-12.608600616455078, 12.058191299438477),
                (-11.36137866973877, 9.659506797790527),
                (-11.284592628479004, 11.02198314666748),
                (-11.390562057495117, 11.181527137756348),
                (-11.062557220458984, 9.009821891784668),
                (-13.004243850708008, 9.230701446533203),
                (-14.876477241516113, 10.149822235107422),
                (-9.741143226623535, 12.573694229125977)]

         # Adjust if you compute real min/max for SVHN
        #min_max_overall = [(-14.876477241516113, 12.573694229125977)]
        # SVHN preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[normal_class][0]] * 3,
                                 [min_max[normal_class][1] - min_max[normal_class][0]] * 3)
        ])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Load train and test datasets
        train_set = MySVHN(root=self.root, split='train', download=True,
                           transform=transform, target_transform=target_transform)
        train_idx_normal = get_target_label_idx(train_set.labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MySVHN(root=self.root, split='test', download=True,
                               transform=transform, target_transform=target_transform)


class MySVHN(SVHN):
    """Torchvision SVHN class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MySVHN, self).__init__(*args, **kwargs)
       # assert self.split in ['train', 'test'], "Split must be 'train' or 'test'"
       # print(f"Split: {self.split}, Data Shape: {self.data.shape}, Labels Shape: {self.labels.shape}")

    def __getitem__(self, index):
        """Override the original method of the SVHN class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """

        if self.split == 'train':
            img, target = self.data[index], self.labels[index]
        elif self.split == 'test':
            img, target = self.data[index], self.labels[index]
        else:
            raise ValueError("Invalid split. Expected 'train' or 'test'.")

        # Convert image to PIL format
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))  # SVHN images are stored as HxWxC

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
