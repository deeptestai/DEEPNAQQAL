import pandas as pd
from scipy.stats import mannwhitneyu

# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv("./llm_t.csv")

# Initialize a list to store results
results = []

# Filter data for AUG -1, 100, 250, and 400
aug_minus_1 = data[data['AUG'] == -1]
aug_100 = data[data['AUG'] == 100]
aug_250 = data[data['AUG'] == 250]
aug_400 = data[data['AUG'] == 400]

# Compare "Acc_Ours" between AUG -1 and AUG 100, 250, 400
model = 'Acc_Ours'

for dataset in data['Dataset'].unique():
    # Filter by dataset
    aug_minus_1_dataset = aug_minus_1[aug_minus_1['Dataset'] == dataset][model].values
    aug_100_dataset = aug_100[aug_100['Dataset'] == dataset][model].values
    aug_250_dataset = aug_250[aug_250['Dataset'] == dataset][model].values
    aug_400_dataset = aug_400[aug_400['Dataset'] == dataset][model].values

    # Define a list of AUG values for comparison
    comparisons = {
        "AUG -1 vs 100": aug_100_dataset,
        "AUG -1 vs 250": aug_250_dataset,
        "AUG -1 vs 400": aug_400_dataset
    }

    for comparison, aug_data in comparisons.items():
        # Ensure there is data for both AUG -1 and the comparison group
        if len(aug_minus_1_dataset) > 1 and len(aug_data) > 1:
            try:
                # Perform Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(aug_minus_1_dataset, aug_data, alternative='two-sided', use_continuity=True)

                # Effect size calculation
                n1, n2 = len(aug_minus_1_dataset), len(aug_data)
                effect_size = (u_stat - (n1 * n2) / 2) / ((n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5)

                significant = "Yes" if p_value < 0.05 else "No"

                # Append results
                results.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Comparison': comparison,
                    'U_Statistic': u_stat,
                    'P_Value': p_value,
                    'Effect_Size': effect_size,
                    'Significant': significant
                })

                print(f"Dataset: {dataset}, Model: {model}, Comparison: {comparison}, U-Statistic: {u_stat}, P-Value: {p_value}, Effect Size: {effect_size}, Significant: {significant}")

            except ValueError as e:
                print(f"Error for Dataset: {dataset}, Model: {model}, Comparison: {comparison}: {e}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_file = "mwu_aug_minus1_vs_all_results_llm.csv"
results_df.to_csv(results_file, index=False)

print(f"Results saved to {results_file}")
