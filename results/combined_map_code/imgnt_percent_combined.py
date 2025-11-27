import os
import re
import pandas as pd

base_dir = "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/results/imagenet/results_remove/img_inc"

output_rows = []

# Extract all AUG-1 accuracies first
aug_data = {}   # { seed_number : { percentage : ours_acc } }

for fname in os.listdir(base_dir):
    if "AUG-1" in fname and fname.endswith(".txt"):
        seed_number = re.findall(r"\d+", fname)
        seed_number = seed_number[0] if seed_number else "Unknown"

        aug_path = os.path.join(base_dir, fname)
        aug_data[seed_number] = {}

        current_percent = None

        with open(aug_path, "r") as f:
            for line in f:
                # Match training percentage
                mp = re.search(r"Training Percentage:\s*(\d+)%", line)
                if mp:
                    current_percent = mp.group(1)

                # Match Acc line
                if line.strip().startswith("Acc") and current_percent:
                    vals = re.findall(r"(\d+\.\d+)", line)
                    if vals:
                        aug_data[seed_number][current_percent] = vals[0]  # Ours accuracy
                        current_percent = None


# Extract results_XX_percent.txt files
for fname in os.listdir(base_dir):
    if fname.startswith("results_") and fname.endswith(".txt"):
        percent = re.findall(r"\d+", fname)
        percent = percent[0] if percent else "Unknown"

        training_fraction = None
        training_accuracy = None

        fpath = os.path.join(base_dir, fname)

        with open(fpath, "r") as f:
            for line in f:
                if "Training Fraction:" in line:
                    m = re.search(r"(\d+)%", line)
                    training_fraction = m.group(1) if m else None

                if "Average Accuracy:" in line:
                    m = re.search(r"(\d+\.\d+)", line)
                    training_accuracy = m.group(1) if m else None

        # Match each results file to each seed's AUG-1 data
        for seed_number, perc_dict in aug_data.items():
            acc_ours = perc_dict.get(percent, None)

            output_rows.append([
                "imagenet",              # Dataset
                "imagenet_baseline",     # Model
                seed_number,             # Seed
                "-1",                    # AUG
                percent,                 # Training Percentage
                acc_ours,                # Ours accuracy
                training_fraction,       # Training Fraction
                training_accuracy        # Training Accuracy
            ])


# Create DataFrame
df = pd.DataFrame(output_rows, columns=[
    "Dataset","Model","Seed","AUG","Training Percentage",
    "Acc_ours","Training Fraction","Training Accuracy"
])

# Convert to numeric for sorting
df["Training Percentage"] = df["Training Percentage"].astype(int)
df["Training Fraction"] = df["Training Fraction"].astype(int)
df["Seed"] = df["Seed"].astype(int)

# SORT BY Training Fraction FIRST, THEN Seed
df = df.sort_values(by=["Training Fraction", "Seed"])

# Save final sorted file
df.to_csv("final_results_imagenet_percent.csv", index=False)

print("Saved: final_sorted_imagenet_results.csv (sorted output)")
