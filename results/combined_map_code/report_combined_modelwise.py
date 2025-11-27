import os
import pandas as pd
import re

# Define base paths
base_path = "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/results"

datasets = {
    "mnist": ["mnist/results_remove/lenettransfer", "mnist/results_remove/vgg16"],
    "svhn": ["svhn/results_remove/svhntransfer", "svhn/results_remove/vgg16"],
    "imagenet": ["imagenet/results_remove/test_llm"]
}

print(f"Using absolute base path: {base_path}")

# Initialize DataFrame
columns = ["Dataset", "Model", "Subfolder", "SEED", "AUG", 
           "TP_Ours", "TN_Ours", "FP_Ours", "FN_Ours", "Acc_Ours", 
           "TP_DAIV", "TN_DAIV", "FP_DAIV", "FN_DAIV", "Acc_DAIV",
           "TP_SO", "TN_SO", "FP_SO", "FN_SO", "Acc_SO",
           "TP_DeepSVDD", "TN_DeepSVDD", "FP_DeepSVDD", "FN_DeepSVDD", "Acc_DeepSVDD", "Folder"]

results_df = pd.DataFrame(columns=columns)

# Function to extract values
def extract_values_from_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    try:
        metrics = {}
        patterns = {
            "TP_Ours": r"TP\s*=\s*(\d+)\s*\|",
            "TN_Ours": r"TN\s*=\s*(\d+)\s*\|",
            "FP_Ours": r"FP\s*=\s*(\d+)\s*\|",
            "FN_Ours": r"FN\s*=\s*(\d+)\s*\|",
            "Acc_Ours": r"Acc\s*=\s*([\d.]+)\s*\|",
            "TP_DAIV": r"TP\s*=\s*\d+\s*\|\s*(\d+)\s*\|",
            "TN_DAIV": r"TN\s*=\s*\d+\s*\|\s*(\d+)\s*\|",
            "FP_DAIV": r"FP\s*=\s*\d+\s*\|\s*(\d+)\s*\|",
            "FN_DAIV": r"FN\s*=\s*\d+\s*\|\s*(\d+)\s*\|",
            "Acc_DAIV": r"Acc\s*=\s*[\d.]+\s*\|\s*([\d.]+)\s*\|",
            "TP_SO": r"TP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|",
            "TN_SO": r"TN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|",
            "FP_SO": r"FP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|",
            "FN_SO": r"FN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|",
            "Acc_SO": r"Acc\s*=\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|",
            
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            metrics[key] = int(match.group(1)) if match and "Acc" not in key else float(match.group(1)) if match else None
        return list(metrics.values())
    except Exception as e:
        print(f"Error parsing file content: {file_path}. Details: {e}")
        return None

# Scan folders and extract data
for dataset, subfolders in datasets.items():
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_path, subfolder)
        folder_name = os.path.basename(subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing dataset: {dataset}, subfolder: {folder_name}")
            for file in sorted(os.listdir(subfolder_path), key=lambda x: (int(re.findall(r'AUG(-?\d+)', x)[0]) if re.findall(r'AUG(-?\d+)', x) else float('inf'), x)):

                if file.endswith(".txt"):
                    print(f"Found file: {file}")
                    seed, aug = re.findall(r"SEED(\d+)_AUG(-?\d+).txt", file)[0]
                    values = extract_values_from_file(os.path.join(subfolder_path, file))
                    if values:
                        results_df.loc[len(results_df)] = [dataset, dataset, subfolder, seed, aug] + values + [folder_name]
                    else:
                        print(f"Skipping file {file}: No values extracted")
        else:
            print(f"Path not found: {subfolder_path}")

# Save results
if not results_df.empty:
    results_df.to_csv("results_summary_updated_llm.csv", index=False)
    print("Results saved to results_summary_updated.csv")
else:
    print("No data collected. Please check file paths and content.")
