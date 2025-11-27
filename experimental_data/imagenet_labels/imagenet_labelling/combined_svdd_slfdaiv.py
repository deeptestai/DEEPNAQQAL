import pandas as pd

# File paths
base_file_path = "./data.csv"
validation_file_path = "./validation_results_imgnt_deepsvdd.csv"
output_file_path = "data1.csv"

# Read the CSV files
base_df = pd.read_csv(base_file_path)
validation_df = pd.read_csv(validation_file_path)

# Extract model name and number from the SAMPLE column in the validation file
validation_df['model_number'] = validation_df['SAMPLE'].str.extract(r'(^\w+_\d+)')

# Rename validation columns for merging
validation_df.rename(columns={"ID/OOD": "ID/OOD DeepSVDD", "score": "Svdd scores"}, inplace=True)

# Add a new column to the base file for matching
base_df['model_number'] = base_df['file'].str.extract(r'(^\w+_\d+)')

# Perform left join based on the extracted model number
merged_df = pd.merge(base_df, validation_df[['model_number', 'ID/OOD DeepSVDD', 'Svdd scores']], on='model_number', how='left')

# Drop the helper column after merging
merged_df.drop(columns=['model_number'], inplace=True)

# Save the combined DataFrame
merged_df.to_csv(output_file_path, index=False)

print(f"Combined file saved as: {output_file_path}")
