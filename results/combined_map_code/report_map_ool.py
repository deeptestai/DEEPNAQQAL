import os
import pandas as pd

# Base directory containing the folders
base_dir = "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/results/imagenet/results_remove/test_ool2_70"

# List of seeds to process
seeds = ["1304", "4020", "230302", "8040", "560602", "350502","1200"]

# Augmentations to process
augmentations = [ "100"]

# List to store extracted data
data = []

# Loop through each subfolder in results_ool_dataset
for dataset_folder in os.listdir(base_dir):
    dataset_path = os.path.join(base_dir, dataset_folder)

    if os.path.isdir(dataset_path):
        # Loop through each file in the dataset folder
        for txt_file in os.listdir(dataset_path):
            if txt_file.endswith(".txt"):
                # Skip files containing 'TEST' in their names for the ImageNet folder
                if "imagenet" in dataset_folder.lower() and "TEST" in txt_file:
                    continue

                # Extract seed, augmentation, and model from the filename
                file_parts = txt_file.replace(".txt", "").split("_")
                seed = file_parts[0].replace("SEED", "")
                aug = file_parts[1].replace("AUG", "")
                model_name = file_parts[-1]  # Model is always the last part

                if seed not in seeds or aug not in augmentations:
                    continue

                # Initialize confusion matrix values
                TP, TN, FP, FN, Acc_Ours, Acc_DAIV, Acc_SO, Acc_llm = None, None, None, None, None, None, None, None

                # Read the file and extract confusion matrix data
                file_path = os.path.join(dataset_path, txt_file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("TP"):
                            TP = line.split("=")[1].strip().split()[0]
                        elif line.startswith("TN"):
                            TN = line.split("=")[1].strip().split()[0]
                        elif line.startswith("FP"):
                            FP = line.split("=")[1].strip().split()[0]
                        elif line.startswith("FN"):
                            FN = line.split("=")[1].strip().split()[0]
                        elif "Acc   =" in line:
                            # Extract accuracy values from the line
                            acc_values = line.split("=")[1].strip().split("|")
                            Acc_Ours = acc_values[0].strip()
                            Acc_DAIV = acc_values[1].strip()
                            Acc_SO = acc_values[2].strip()
                            Acc_DeepSVDD = acc_values[3].strip()
                            Acc_llm = acc_values[4].strip()
                            Acc_llm2 = acc_values[5].strip()
                            Acc_llm3 = acc_values[6].strip()


                # Append data to the list
                data.append({
                    "Dataset": dataset_folder,
                    "Seed": seed,
                    "Augmentation": aug,
                    "Model": model_name,  # Include model name (e.g., dlf, ox, etc.)
                    "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "Acc_Ours": Acc_Ours,
                    "Acc_DAIV": Acc_DAIV,
                    "Acc_SO": Acc_SO,
                    "Acc_DeepSVDD":Acc_DeepSVDD,
                    "Acc_llm": Acc_llm,
                    "Acc_llm2": Acc_llm2,
                    "Acc_llm3": Acc_llm3
                })

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Ensure proper ordering: Augmentation first, then Seed, then Dataset
df['Augmentation'] = pd.Categorical(df['Augmentation'], categories=augmentations, ordered=True)
df['Seed'] = pd.Categorical(df['Seed'], categories=seeds, ordered=True)

# Sorting Logic
# - Dataset-wise: mnist -> svhn -> imagenet
# - Augmentation: -1 -> 100 -> 250 -> 400
# - Model-wise for each seed: Group all models (dlf, dx, etc.) for each seed
df = df.sort_values(by=['Dataset', 'Augmentation', 'Model', 'Seed'])

# Save to a CSV file
output_file = "compiled_results_ool2_rq3_model_aug.csv"
df.to_csv(output_file, index=False)

print(f"Data extracted and saved to {output_file}.")
