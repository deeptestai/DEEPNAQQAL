import os
import re
import pandas as pd

# Path to your folder
folder = "/home/vincenzo.riccio/human-feedback-validity-checker-dnn/results/imagenet/results_remove/test_ool2_70"

rows = []

for filename in os.listdir(folder):
    if not filename.endswith(".txt"):
        continue
    
    # ---- Extract seed, augmentation, model from filename ----
    # Example: SEED1304_AUG100_dlf.txt
    match = re.match(r"SEED(\d+)_AUG(\d+)_(\w+)\.txt", filename)
    if not match:
        continue
    
    seed, aug, model = match.groups()
    
    # ---- Read the file ----
    with open(os.path.join(folder, filename), "r") as f:
        text = f.read()
    
    # ---- Extract accuracy line ----
    # Example: Acc   =   0.773   |  0.773   |  0.761   |  0.761   |  0.477   |  0.773   |  0.705
    acc_line = re.search(r"Acc\s*=\s*(.*)", text)
    if not acc_line:
        continue
    
    # Split columns
    values = [v.strip() for v in acc_line.group(1).split("|")]
    
    # Convert to float
    values = [float(v) for v in values]
    
    # ---- Build output row ----
    row = {
        "Dataset": "imagenet",
        "Mode": "remove",
        "validator": "vgg16_transfer",
        "Seed": int(seed),
        "Augmentation": int(aug),
        "Model": model,
        "Acc_Ours": values[0],
        "Acc_DAIV": values[1],
        "Acc_SO": values[2],
        "Acc_DeepSVDD": values[3],
        "Acc_LLM": values[4],
        "Acc_LLM2": values[5],
        "Acc_LLM3": values[6]
        # LLM3 exists but unused in your desired format
    }
    
    rows.append(row)

# ---- Create DataFrame ----
df = pd.DataFrame(rows)

# ---- Sort for readability ----
df = df.sort_values(by=["Model", "Seed"])

# ---- Save CSV ----
df.to_csv("RQ3_results.csv", index=False)

print(df)
