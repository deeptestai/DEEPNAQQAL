import os
import pandas as pd
import re

# --------------------------- CONFIG -------------------------------

base_path = "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/results"

datasets = ["imagenet"]
subfolders = ["results_remove/test_imagenet"]

print(f"Using base path: {base_path}")

validators = ["Ours","DAIV","SO","DeepSVDD","LLM","LLM2","LLM3"]

# -------------------- DataFrame columns ----------------------------

columns = ["Dataset", "Model", "SEED", "AUG"]

for metric in ["TP", "TN", "FP", "FN", "Acc"]:
    for v in validators:
        columns.append(f"{metric}_{v}")

results_df = pd.DataFrame(columns=columns)

# ---------------------- PARSING HELPERS ----------------------------

def extract_row(line, metric_name, is_accuracy=False):
    """
    Extract values from a line like:
    'TP  =   5 | 1 | 0 | 0 | 7 | 2 | 7'
    """
    if not line.strip().startswith(metric_name):
        return None

    # Split at "=" to get only the numbers part
    _, right = line.split("=", 1)

    # Split by pipes
    parts = [p.strip() for p in right.split("|")]

    # Must have exactly 7 validators
    if len(parts) != 7:
        return None

    # Convert
    if is_accuracy:
        return [float(p) for p in parts]
    else:
        return [int(p) for p in parts]


def extract_values_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    metrics = {}

    for line in lines:

        # TP
        row = extract_row(line, "TP")
        if row:
            for i, v in enumerate(validators):
                metrics[f"TP_{v}"] = row[i]

        # TN
        row = extract_row(line, "TN")
        if row:
            for i, v in enumerate(validators):
                metrics[f"TN_{v}"] = row[i]

        # FP
        row = extract_row(line, "FP")
        if row:
            for i, v in enumerate(validators):
                metrics[f"FP_{v}"] = row[i]

        # FN
        row = extract_row(line, "FN")
        if row:
            for i, v in enumerate(validators):
                metrics[f"FN_{v}"] = row[i]

        # Accuracy
        row = extract_row(line, "Acc", is_accuracy=True)
        if row:
            for i, v in enumerate(validators):
                metrics[f"Acc_{v}"] = row[i]

    return metrics


# ------------------------ MAIN LOOP ------------------------------

for dataset in datasets:
    for subfolder in subfolders:

        path = os.path.join(base_path, dataset, subfolder)

        if not os.path.isdir(path):
            print(f"Path not found: {path}")
            continue

        print(f"Processing dataset: {dataset}, folder: {subfolder}")

        for file in os.listdir(path):
            if not file.endswith(".txt"):
                continue

            try:
                seed, aug = re.findall(r"SEED(\d+)_AUG(-?\d+)\.txt", file)[0]

                file_path = os.path.join(path, file)
                metrics = extract_values_from_file(file_path)

                row = {
                    "Dataset": dataset,
                    "Model": dataset,
                    "SEED": int(seed),
                    "AUG": int(aug)
                }

                row.update(metrics)
                results_df.loc[len(results_df)] = row

            except Exception as e:
                print(f"Skipping {file}: {e}")


# -------------------------- SAVE ---------------------------------

results_df = results_df.sort_values(by=["Dataset", "AUG", "SEED"])
results_df.to_csv("results_summary_rq1.csv", index=False)

print("✅ Results saved to results_summary_final.csv")
