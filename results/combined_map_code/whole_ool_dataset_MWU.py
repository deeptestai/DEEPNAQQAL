import pandas as pd
from scipy.stats import mannwhitneyu

# Load dataset
file_path = "RQ3_results.csv"  # Replace with actual file path
df = pd.read_csv(file_path)

# Group datasets
datasets = df["Dataset"].unique()

# Store results
results = []

# Function to compute Mann-Whitney U test
def perform_stat_test(best_acc, compare_acc):
    if len(best_acc) == 0 or len(compare_acc) == 0:
        return None, None, None, "No Data"

    u_stat, p_value = mannwhitneyu(best_acc, compare_acc, alternative='two-sided', use_continuity=True)
    n1, n2 = len(best_acc), len(compare_acc)

    # Effect size calculation
    effect_size = (u_stat - (n1 * n2) / 2) / ((n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5)

    # Significance check
    significant = "Yes" if p_value < 0.05 else "No"

    return u_stat, p_value, effect_size, significant

# Iterate through each dataset
for dataset in datasets:
    df_subset = df[df["Dataset"] == dataset]

    # Identify the best and other accuracy columns
    acc_columns = ["Acc_Ours", "Acc_DAIV", "Acc_SO", "Acc_DeepSVDD", "Acc_llm", "Acc_llm2","Acc_llm3"]
    means = df_subset[acc_columns].mean().sort_values(ascending=False)

    best_col = means.index[0]
    best_acc = df_subset[best_col].values

    # Iterate through second-best, third-best, etc., until a significant result is found
    for i in range(1, len(means)):
        compare_col = means.index[i]
        compare_acc = df_subset[compare_col].values

        # Perform Mann-Whitney U Test
        u_stat, p_value, effect_size, significant = perform_stat_test(best_acc, compare_acc)

        results.append({
            "Dataset": dataset,
            "Best_Accuracy": best_col,
            "Compared_Accuracy": compare_col,
            "Mann_Whitney_U": round(u_stat, 4) if u_stat is not None else None,
            "p_value": round(p_value, 6) if p_value is not None else None,
            "Effect_Size": round(effect_size, 4) if effect_size is not None else None,
            "Significant": significant
        })

        # Stop if a significant result is found
        if significant == "Yes":
            break

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
output_file = "MWU_best_vs_until_significant_ool2imgrq3_wholeavg.csv"
results_df.to_csv(output_file, index=False)

# Display results
output_file
