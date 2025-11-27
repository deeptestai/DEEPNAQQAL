import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

# Load your dataset
df = pd.read_csv("final_results_imagenet_percent.csv")

# Convert to numeric types
df["Training Fraction"] = df["Training Fraction"].astype(int)
df["Training Accuracy"] = df["Training Accuracy"].astype(float)
df["Acc_ours"] = df["Acc_ours"].astype(float)

# Reference: Acc_ours at 100%
acc_ours_100 = df[df["Training Fraction"] == 100]["Acc_ours"].values

# Cohen's d function
def cohens_d(x, y):
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_sd

results = []

# Perform tests for 90 → 10 in your desired order
for p in [90, 80, 70, 60, 50, 40, 30, 20, 10]:
    train_acc_p = df[df["Training Fraction"] == p]["Training Accuracy"].values
    
    # Mann–Whitney U (two-sided)
    u_stat, p_val = mannwhitneyu(train_acc_p, acc_ours_100, alternative="two-sided")
    
    # Cohen's d
    d_val = cohens_d(train_acc_p, acc_ours_100)
    
    # Significance
    sig = "Yes" if p_val < 0.05 else "No"
    
    # Append results
    results.append([
        f"{p}% vs 100%",
        round(u_stat, 4),
        round(p_val, 6),
        round(d_val, 4),
        sig
    ])

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=[
    "Comparison", "U_statistic", "p_value", "Cohen_d", "Significant"
])

# Print nicely
print(results_df.to_string(index=False))

# ❗ Save to CSV
results_df.to_csv("mann_whitney_results_for_percent.csv", index=False)

print("\nSaved results to mann_whitney_results.csv")
