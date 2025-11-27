import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def compute_and_display_difference(image_path1, image_path2, amplify=10):
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert images to grayscale for SSIM computation
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    ssim_score, _ = ssim(gray1, gray2, full=True)

    # Compute mean pixel difference
    mean_pixel_diff = np.mean(np.abs(image1.astype("float") - image2.astype("float")))

    # Compute absolute difference and amplify differences for visibility
    difference_image = cv2.absdiff(image1, image2)
    difference_image = cv2.convertScaleAbs(difference_image, alpha=amplify)

    # Display images and their difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(difference_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Difference Image (Amplified by {amplify}x)")
    axes[2].axis("off")

    # Print similarity metrics
    print(f"SSIM Score: {ssim_score:.4f} (Closer to 1.0 means more similar)")
    print(f"Mean Pixel Difference: {mean_pixel_diff:.4f} (Lower value means more similar)")

    plt.show()


# Paths to the images
image_path1 = "image_test/dlfuzz_2_pred-452.png"
image_path2 = "image_test/dlfuzz_4_pred-452.png"

# Run the comparison and display
compute_and_display_difference(image_path1, image_path2, amplify=10)
