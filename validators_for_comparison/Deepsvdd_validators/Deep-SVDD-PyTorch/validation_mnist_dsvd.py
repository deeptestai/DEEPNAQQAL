import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from deepSVDD import DeepSVDD
from datasets.preprocessing import global_contrast_normalization
import sys

# Add the src directory to the Python path
sys.path.append('./Deep-SVDD-PyTorch/src')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Assume `normal_class` is the class you are testing for
min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

# Validation preprocessing pipeline
transform = transforms.Compose([
            #transforms.Resize((224, 224)),  # Resize all images to 224x224
            #transforms.ToTensor(),
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize(
                [min_max[5][0]],
                [min_max[5][1] - min_max[5][0]]
            )
        ])

def validate_and_save_results(root_dir, model_ckpt_path, threshold,objective, output_csv):
    """
    Validate .npy files in subfolders and save results in a CSV file.

    Args:
        root_dir (str): Path to the root directory containing subfolders (e.g., 'deepjanus', 'dx', 'sv').
        model_ckpt_path (str): Path to the Deep SVDD model checkpoint.
        objective (str): Deep SVDD objective ('one-class' or 'soft-boundary').
        threshold (float): Anomaly score threshold for validity.
        output_csv (str): Path to save the output CSV file.
    """
    # Initialize Deep SVDD
    net = DeepSVDD(objective, 0.1)
    net.set_network('mnist_LeNet')  # Update network name if needed
    net.load_model(model_path=model_ckpt_path, load_ae=True)

    with open(output_csv, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(['TOOL', 'SAMPLE', 'ID/OOD', 'score'])  # CSV Header

        # Iterate through subfolders
        for subdir, _, files in os.walk(root_dir):
            tool_name = os.path.basename(subdir)
            print(f"Processing folder: {tool_name}")

            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(subdir, file)
                    # Load .npy file
                    sample = np.load(file_path)
                    sample = sample * 255
                    print("Sample dtype:", sample.dtype)
                    print("Sample range: min =", sample.min(), ", max =", sample.max())

            	    # Directly convert the sample to a tensor without transformations
            	    # Convert to PyTorch tensor
                    sample = torch.tensor(sample, dtype=torch.float32)
                   # sample = sample.permute(2, 0, 1)  # Reorder dimensions to [Channels, Height, Width] if needed
                    sample = sample.squeeze(0)  
                    # Check if resizing or reshaping is needed
                    #sample = sample.squeeze(-1) #for mnist
                    sample = sample.permute(2, 0, 1)
                    print("Sample shape before preprocessing:", sample.shape)

                    # Apply transformations
                    inputs = transform(sample).unsqueeze(0) 
                    print("with GCN after mul 255: min =", inputs.min(), ", max =", inputs.max())
                    print("Transformed sample shape:", inputs.shape)


                    # Run through DeepSVDD model
                    outputs = net.single_tests(inputs, device)
                    dist = torch.sum((outputs.squeeze().data.cpu() - torch.tensor(net.c)) ** 2)
                    score = dist - net.R ** 2
                    print("Radius (R):", net.R)
                    # print("Center (c):", net.c)

                    # Determine ID/OOD
                    id_ood = "ood" if score > threshold else "id"

                    # Write result to CSV
                    writer.writerow([tool_name, file, id_ood, score.item()])

    print(f"Validation complete. Results saved to {output_csv}.")

if __name__ == '__main__':
    root_dir = './generated_images/mnist_inputs'  # Root directory containing subfolders
    ckpt_path = './Deep-SVDD-PyTorch/log/mnist_test/deepSVDD_ckpt_soft_boundary_mnist'  # Model checkpoint
    output_csv = './validation_results_mnist_deepsvdd.csv'  # Output CSV file
    objective = 'soft-boundary'
    threshold = 0.17415527193755853    #0.09370872959331056                                                     # 0.0026523620557009173 Imagenet                       # Set anomaly score threshold

    validate_and_save_results(root_dir, ckpt_path, threshold, objective, output_csv)

