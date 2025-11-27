import pandas as pd
from scipy.stats import mannwhitneyu

# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv("./compiled_results_ool70_clustered_model_aug.csv")

# Initialize a list to store results
results = []

# Group by Dataset and Model to apply the Mann-Whitney U test
for (dataset, Model), group in data.groupby(['Dataset', 'Model']):
    # Extract relevant columns
    ours = group['Acc_Ours'].values
    daiv = group['Acc_DAIV'].values
    so = group['Acc_SO'].values
    deepsvdd = group['Acc_DeepSVDD'].values

    # Debugging: Print the data being compared
    print(f"Dataset: {dataset}, Model: {Model}")
    print(f"Acc_Ours: {ours}")
    print(f"Acc_DAIV: {daiv}")
    print(f"Acc_SO: {so}")
    print(f"Acc_DeepSVDD: {deepsvdd}")

    # Determine the best and second-best models based on mean accuracy
    mean_accuracies = {
        'Ours': ours.mean(),
        'DAIV': daiv.mean(),
        'SO': so.mean(),
        'DeepSVDD': deepsvdd.mean()
    }
    
    # Sort the models based on mean accuracy
    sorted_models = sorted(mean_accuracies.items(), key=lambda x: x[1], reverse=True)

    best_model, second_best_model = sorted_models[0], sorted_models[1]

    # Select corresponding accuracy arrays
    best_acc = group[f"Acc_{best_model[0]}"].values
    second_best_acc = group[f"Acc_{second_best_model[0]}"].values

    print(f"Comparing {best_model[0]} vs {second_best_model[0]}")
    print(f"Best Acc: {best_acc}, Second Best Acc: {second_best_acc}")

    # Ensure both groups have enough data points for statistical testing
    if len(best_acc) > 1 and len(second_best_acc) > 1:
        try:
            # Mann-Whitney U test with continuity correction to handle ties
            u_stat, p_value = mannwhitneyu(best_acc, second_best_acc, alternative='two-sided', use_continuity=True)
            n1, n2 = len(best_acc), len(second_best_acc)

            # Effect size calculation
            effect_size = (u_stat - (n1 * n2) / 2) / ((n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5)

            significant = "Yes" if p_value < 0.05 else "No"

            results.append({
                'Dataset': dataset,
                'Model': Model,
                'Comparison': f'{best_model[0]} vs {second_best_model[0]}',
                'U_Statistic': u_stat,
                'P_Value': p_value,
                'Effect_Size': effect_size,
                'Significant': significant
            })

            print(f"U-Statistic: {u_stat}, P-Value: {p_value}, Effect Size: {effect_size}, Significant: {significant}")

        except ValueError as e:
            print(f"Error for Dataset: {dataset}, Model: {Model}: {e}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_file = "mwu_results_with_best_and_second_best_ool.csv"
results_df.to_csv(results_file, index=False)

print(f"Mann-Whitney U test results saved to '{results_file}'.")
