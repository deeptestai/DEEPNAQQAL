from PIL import Image
import torch
import numpy as np
import os
from torchvision.transforms import Resize

def global_contrast_normalization(x: torch.Tensor, scale='l2'):
    """
    Apply Global Contrast Normalization to the tensor.
    """
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)  # mean across all features
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    elif scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale
    return x

# Path to ImageNet dataset
path_to_images = "/home/vincenzo.riccio/Deep-SVDD-PyTorch/src/data/MyImgNt/train_imagenet/"
resize = Resize((224, 224))  # Resize to 224x224

# Check if the main directory exists
if not os.path.exists(path_to_images):
    raise ValueError(f"Path {path_to_images} does not exist!")

# Iterate through each class subfolder
for class_folder in os.listdir(path_to_images):
    class_folder_path = os.path.join(path_to_images, class_folder)

    # Ensure the path is a directory
    if os.path.isdir(class_folder_path):
        print(f"\nProcessing class folder: {class_folder}")

        min_vals, max_vals = [], []  # Reset min/max for this class

        # Iterate through images in the class folder
        for img_file in os.listdir(class_folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_folder_path, img_file)

                try:
                    # Open and preprocess the image
                    img = Image.open(img_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")  # Convert to RGB if not already
                    img = resize(img)
                    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
                    gcn_img = global_contrast_normalization(img_tensor, scale='l2')

                    # Collect min and max values
                    min_vals.append(gcn_img.min().item())
                    max_vals.append(gcn_img.max().item())

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        # Ensure that we processed some images for this folder
        if min_vals and max_vals:
            print(f"Class {class_folder} Min: {min(min_vals)}")
            print(f"Class {class_folder} Max: {max(max_vals)}")
        else:
            print(f"No valid images were processed for class {class_folder}.")
