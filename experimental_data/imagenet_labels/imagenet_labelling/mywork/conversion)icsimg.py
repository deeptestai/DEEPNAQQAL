import os
import numpy as np
import cv2
from PIL import Image

# Define paths
input_dir = "/home/maryam/Documents/human-feedback-validity-checker-dnn/label/imagenet_labelling/imagenet_inputs"
output_dir = "/home/maryam/Documents/human-feedback-validity-checker-dnn/label/imagenet_labelling/imagenet_output"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def normalize_image(img):
    """
    Normalize image values from arbitrary ranges to [0, 255] for consistent brightness.
    """
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min) * 255  # Scale to [0, 255]
    return np.clip(img, 0, 255).astype(np.uint8)

def process_image(image_data, npy_file):
    """
    Process the image by normalizing and converting to RGB format.
    """
    # Step 1: Normalize the image values to [0, 255]
    img = normalize_image(image_data)

    # Step 2: Handle channel configuration (channels-first to channels-last)
    if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # Channels-first format (C, H, W)
        img = np.transpose(img, (1, 2, 0))  # Convert to HWC (Height, Width, Channels)

    # Step 3: Convert grayscale to RGB if needed
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR format for color consistency

    return img

# Process each .npy file and convert to .png
for npy_file in os.listdir(input_dir):
    if npy_file.endswith(".npy"):
        try:
            # Load the .npy file
            image_data = np.load(os.path.join(input_dir, npy_file))[0]  # Load the first element if it's a batch

            # Process the image
            img = process_image(image_data, npy_file)

            # Save as PNG
            output_path = os.path.join(output_dir, f"{npy_file.replace('.npy', '')}.png")
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"Error processing file {npy_file}: {e}")

print("All .npy images have been processed. Check the output directory.")