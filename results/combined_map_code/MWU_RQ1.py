import pandas as pd
from scipy.stats import mannwhitneyu

# Load the dataset (replace 'your_file.csv' with the actual file path)
data = pd.read_csv("./RQ1-summary2.csv")

# Initialize a list to store results
results = []

# Group by Dataset, Validator, and AUG to apply the Mann-Whitney U test
for (dataset, validator, aug), group in data.groupby(['Dataset', 'validator', 'AUG']):
    # Extract relevant columns
    accuracy_dict = {
        'Ours': group['Acc_Ours'].values,
        'DAIV': group['Acc_DAIV'].values,
        'SO': group['Acc_SO'].values,
        'DeepSVDD': group['Acc_DeepSVDD'].values,
        'LLM': group['Acc_LLM'].values,
        'LLM2': group['Acc_LLM2'].values,
        'LLM3': group['Acc_LLM3'].values
    }
    
    # Ensure at least one model has valid accuracy values
    mean_accuracies = {model: acc.mean() for model, acc in accuracy_dict.items() if len(acc) > 1}
    if not mean_accuracies:
        print(f"Warning: No valid accuracy values for Dataset: {dataset}, Validator: {validator}, AUG: {aug}. Skipping.")
        continue  # Skip this iteration
    
    sorted_models = sorted(mean_accuracies.items(), key=lambda x: x[1], reverse=True)
    best_model = sorted_models[0][0]
    best_acc = accuracy_dict[best_model]
    
    print(f"Dataset: {dataset}, Validator: {validator}, AUG: {aug}, Best Model: {best_model}, Acc: {best_acc}")
    
    # Handle special case where ImageNet has only one validator
    if dataset == "imagenet":
        if len(sorted_models) < 2:
            print(f"Skipping ImageNet dataset with only one available model: {validator}")
            continue
    
    # Iterate through second-best, third-best, etc., until a significant result is found
    for i in range(1, len(sorted_models)):
        compare_model = sorted_models[i][0]
        compare_acc = accuracy_dict[compare_model]
        
        print(f"Comparing {best_model} vs {compare_model}")
        
        if len(best_acc) > 1 and len(compare_acc) > 1:
            try:
                u_stat, p_value = mannwhitneyu(best_acc, compare_acc, alternative='two-sided', use_continuity=True)
                n1, n2 = len(best_acc), len(compare_acc)
                
                # Effect size calculation
                effect_size = (u_stat - (n1 * n2) / 2) / ((n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5)
                
                significant = "Yes" if p_value < 0.05 else "No"
                
                results.append({
                    'Dataset': dataset,
                    'Validator': validator,
                    'AUG': aug,
                    'Comparison': f'{best_model} vs {compare_model}',
                    'U_Statistic': u_stat,
                    'P_Value': p_value,
                    'Effect_Size': effect_size,
                    'Significant': significant
                })
                
                print(f"U-Statistic: {u_stat}, P-Value: {p_value}, Effect Size: {effect_size}, Significant: {significant}")
                
                # Stop if a significant result is found
                if significant == "Yes":
                    break
            except ValueError as e:
                print(f"Error for Dataset: {dataset}, Validator: {validator}, AUG: {aug}, Comparison: {best_model} vs {compare_model}: {e}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_file = "mwu_results_rq1_validators2.csv"
results_df.to_csv(results_file, index=False)

print(f"Mann-Whitney U test results saved to '{results_file}'.")
