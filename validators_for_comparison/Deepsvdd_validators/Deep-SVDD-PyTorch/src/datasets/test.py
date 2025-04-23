from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

class MyImgNt(ImageFolder):
    """Custom ImageFolder class to patch the __getitem__ method."""

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
        # Get the image path and its target (label)
        path, target = self.samples[index]
        
        # Open the image
        img = Image.open(path).convert("RGB")  # Convert to RGB
        
        # Apply transformations if specified
        if self.transform is not None:
            img = self.transform(img)
        
        # Return the image, its label, and the index
        return img, target, index


# Define paths to train and validation datasets
train_path = "/home/vincenzo.riccio/Deep-SVDD-PyTorch/src/data/MyImgNt/train_imagenet"
val_path = "/home/vincenzo.riccio/Deep-SVDD-PyTorch/src/data/MyImgNt/val_imagenet"

# Transformation to apply to images (example: resizing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor()          # Convert to tensor
])

# Load train and validation datasets using the custom MyImgNt class
train_dataset = MyImgNt(root=train_path, transform=transform)
val_dataset = MyImgNt(root=val_path, transform=transform)

# Count the number of samples in each dataset
num_train_samples = len(train_dataset)
num_val_samples = len(val_dataset)

# Display counts
print(f"Number of training samples: {num_train_samples}")
print(f"Number of validation samples: {num_val_samples}")

# Optionally, load a single sample from train/validation datasets to test
img, target, index = train_dataset[0]
print("Train sample:")
print("  Image shape:", img.shape)  # Tensor shape (C, H, W)
print("  Target (label):", target)  # Integer label
print("  Index:", index)

img, target, index = val_dataset[0]
print("Validation sample:")
print("  Image shape:", img.shape)  # Tensor shape (C, H, W)
print("  Target (label):", target)  # Integer label
print("  Index:", index)
