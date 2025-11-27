import pandas as pd

# Load the base file
base_file = 'TIGvalidity - SVHN_SURVEY.csv'
base_df = pd.read_csv(base_file)

# Load the validation results file
validation_file = 'validation_results_svhn_deepsvdd.csv'
validation_df = pd.read_csv(validation_file)

# Extract relevant columns from validation file
validation_df['SAMPLE'] = validation_df['SAMPLE'].str.replace('.npy', '', regex=False)

# Merge the data based on matching 'ID' in base file with 'SAMPLE' in validation file
merged_df = base_df.merge(validation_df, left_on='ID', right_on='SAMPLE', how='left')

# Add the 'DeepSVDD' and 'Svdd scores' columns
merged_df['ID/OOD DeepSVDD'] = merged_df['ID/OOD']
merged_df['Svdd scores'] = merged_df['score']

# Drop unnecessary columns from merge (like 'SAMPLE', if needed)
merged_df.drop(['SAMPLE', 'ID/OOD', 'score'], axis=1, inplace=True)

# Save the updated base file
updated_file = 'TIGvalidity - SVHN_SURVEY.csv'
merged_df.to_csv(updated_file, index=False)

print(f"The updated file has been saved as {updated_file}.")
