import os
import re
import pandas as pd

# Define base paths
base_path = "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/results"

datasets = {
    "mnist": ["mnist/results_remove/test_llm"],
    "svhn": ["svhn/results_remove/test_llm"],
    "imagenet": ["imagenet/results_remove/test_llm"]
}

print(f"Using absolute base path: {base_path}")

# Initialize DataFrame
columns = ["Dataset", "Model", "Subfolder", "SEED", "AUG", 
           "TP_LLM2", "TN_LLM2", "FP_LLM2", "FN_LLM2", "Acc_LLM2", "Folder"]

results_df = pd.DataFrame(columns=columns)

# Function to extract LLM values from file
def extract_llm_values_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        patterns = {
            "TP_LLM2": r"TP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "TN_LLM2": r"TN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FP_LLM2": r"FP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FN_LLM2": r"FN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "Acc_LLM2": r"Acc\s*=\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)"
        }
        
        metrics = []
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = float(match.group(1)) if "Acc" in key else int(match.group(1))
                metrics.append(value)
            else:
                metrics.append(-1)  # Use -1 to indicate missing values explicitly
        
        print(f"Extracted LLM values from {file_path}: {metrics}")  # Debugging print statement
        return metrics
    except Exception as e:
        print(f"Error parsing file content: {file_path}. Details: {e}")
        return [-1] * (len(columns) - 7)

# Scan folders and extract LLM data
for dataset, subfolders in datasets.items():
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_path, subfolder)
        folder_name = os.path.basename(subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing dataset: {dataset}, subfolder: {folder_name}")
            for file in sorted(os.listdir(subfolder_path)):
                if file.endswith(".txt"):
                    print(f"Found file: {file}")
                    match = re.search(r"SEED(\d+)_AUG(-?\d+).txt", file)
                    if match:
                        seed, aug = match.groups()
                        values = extract_llm_values_from_file(os.path.join(subfolder_path, file))
                        print(f"Attempting to insert {len(values)} LLM values into DataFrame.")  # Debugging print statement
                        results_df.loc[len(results_df)] = [dataset, dataset, subfolder, seed, aug] + values + [folder_name]

# Sort results by Dataset and AUG values in order (-1, 100, 250, 400)
results_df["AUG"] = results_df["AUG"].astype(int)
results_df.sort_values(by=["Dataset", "AUG"], ascending=[True, True], inplace=True)

# Save results
if not results_df.empty:
    output_file = os.path.join(base_path, "llm2_results_summary.csv")
    results_df.to_csv(output_file, index=False)
    print(f"LLM results saved to {output_file}")
