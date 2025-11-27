import os
import numpy as np
from PIL import Image
import cv2

# Define paths and normalization values
input_dir = "/home/maryam/Documents/human-feedback-validity-checker-dnn/label/imagenet_labelling/imagenes"  # Directory containing .npy files
output_dir = "/home/maryam/Documents/human-feedback-validity-checker-dnn/label/imagenet_labelling/imagenet_output"  # Directory to save .png files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


def adjust_histogram(img):
    """
    Adjust image brightness and contrast using histogram stretching.
    """
    img_min = np.min(img)
    img_max = np.max(img)

    if img_max > img_min:
        # Apply contrast stretching
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = img * 255  # If image is flat (same values), still convert to uint8

    img = np.clip(img, 0, 255)  # Ensure values are within [0, 255]
    return img.astype(np.uint8)


def custom_rescale_and_adjust(img, npy_file):
    """
    Dynamically adjust pixel values based on min/max range and image quality.
    """
    min_val = img.min()
    max_val = img.max()

    print(f"Processing {npy_file}: min={min_val}, max={max_val}")

    # Strategy 1: If values are in [0, 1], scale up to [0, 255]
    if min_val >= 0 and max_val <= 1:
        print(f"Scaling [0, 1] for {npy_file}")
        img = img * 255

    # Strategy 2: If values are within [-1, 1], normalize to [0, 255]
    elif min_val >= -1 and max_val <= 1:
        print(f"Scaling [-1, 1] for {npy_file}")
        img = ((img - min_val) / (max_val - min_val)) * 255

    # Strategy 3: If values are out of the [0, 255] range, rescale
    elif min_val < 0 or max_val > 255:
        print(f"Rescaling extreme values for {npy_file}")
        img = 255 * (img - min_val) / (max_val - min_val)

    # Apply histogram adjustment for better contrast
    img = adjust_histogram(img)

    return np.clip(img, 0, 255).astype(np.uint8)


def process_image(image_data, npy_file):
    """
    Process the image by adjusting and converting to proper pixel values and format.
    """
    # Step 1: Dynamically rescale and adjust the image
    img = custom_rescale_and_adjust(image_data, npy_file)

    # Step 2: Handle channel configuration (channels-first to channels-last)
    if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # Channels-first format (C, H, W)
        img = np.transpose(img, (1, 2, 0))  # Convert to HWC (Height, Width, Channels)

    # Step 3: Handle grayscale images by converting to RGB
    if img.shape[2] == 1:  # Grayscale image
        img = np.repeat(img, 3, axis=2)  # Repeat grayscale channel to get RGB

    # Step 4: Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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