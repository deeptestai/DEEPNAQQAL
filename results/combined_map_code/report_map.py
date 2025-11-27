import os
import pandas as pd
import re

# Define base paths
base_path = "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/results"

datasets = ["mnist", "svhn", "imagenet"]
subfolders = ["results_remove"]  # Single subfolder for ImageNet
print(f"Using absolute base path: {base_path}")

# Initialize DataFrame
columns = ["Dataset", "Model", "SEED", "AUG", 
           "TP_Ours", "TN_Ours", "FP_Ours", "FN_Ours", "Acc_Ours", 
           "TP_DAIV", "TN_DAIV", "FP_DAIV", "FN_DAIV", "Acc_DAIV",
           "TP_SO", "TN_SO", "FP_SO", "FN_SO", "Acc_SO",
           "TP_DeepSVDD", "TN_DeepSVDD", "FP_DeepSVDD", "FN_DeepSVDDc_DeepSVDD",
           "TP_LLM", "TN_LLM", "FP_LLM", "FN_LLM", "Acc_LLM",
           "TP_LLM2", "TN_LLM2", "FP_LLM2", "FN_LLM2", "Acc_LLM2",
            "TP_LLM3", "TN_LLM3", "FP_LLM3", "FN_LLM3", "Acc_LLM3"]

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

            "TP_DeepSVDD": r"TP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "TN_DeepSVDD": r"TN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FP_DeepSVDD": r"FP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FN_DeepSVDD": r"FN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "Acc_DeepSVDD": r"Acc\s*=\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)"

            "TP_LLM": r"TP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "TN_LLM": r"TN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FP_LLM": r"FP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FN_LLM": r"FN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "Acc_LLM": r"Acc\s*=\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)"

            "TP_LLM2": r"TP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "TN_LLM2": r"TN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FP_LLM2": r"FP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FN_LLM2": r"FN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "Acc_LLM2": r"Acc\s*=\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)"

            "TP_LLM3": r"TP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "TN_LLM3": r"TN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FP_LLM3": r"FP\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "FN_LLM3": r"FN\s*=\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            "Acc_LLM3": r"Acc\s*=\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)"

        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = int(match.group(1)) if "Acc" not in key else float(match.group(1))
            else:
                metrics[key] = None  # Default to None if not found

        return list(metrics.values())

    except Exception as e:
        print(f"Error parsing file content: {file_path}. Details: {e}")
        return None

# Collect results
for dataset in datasets:
    for subfolder in subfolders:
        path = os.path.join(base_path, dataset, subfolder)
        if os.path.isdir(path):
            print(f"Processing {dataset} in {subfolder}")
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    try:
                        seed, aug = re.findall(r"SEED(\d+)_AUG(-?\d+).txt", file)[0]
                        model_label = dataset
                        values = extract_values_from_file(os.path.join(path, file))
                        if values:
                            results_df.loc[len(results_df)] = [dataset, model_label, seed, aug] + values
                    except (IndexError, AttributeError, ValueError) as e:
                        print(f"Skipping file {file}: {e}")
        else:
            print(f"Path not found: {path}")

# Sort and rearrange results by dataset order
results_df["SEED"] = results_df["SEED"].astype(int)
results_df["AUG"] = results_df["AUG"].astype(int)
results_df = results_df.sort_values(by=["Dataset", "AUG", "SEED"], key=lambda col: col.map(lambda x: ["mnist", "svhn", "imagenet"].index(x) if x in ["mnist", "svhn", "imagenet"] else x))

# Save results
results_df.to_csv("results_summaryall_updated.csv", index=False)
print("Results saved to results_summary_updated.csv")
