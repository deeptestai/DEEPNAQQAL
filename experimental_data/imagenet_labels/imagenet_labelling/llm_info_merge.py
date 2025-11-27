import pandas as pd

# Load the first file
file1_path = 'data2.csv'  # Update with actual path
file1 = pd.read_csv(file1_path)

# Load the second file
file2_path = 'merged_llama2_classifications.csv'  # Update with actual path
file2 = pd.read_csv(file2_path)

# Normalize column names to avoid case sensitivity issues
file1.columns = file1.columns.str.strip()
file2.columns = file2.columns.str.strip()

# Remove ".npy" extension from Base_ID in file1
file1['Base_ID'] = file1['file'].str.replace(r'\.npy$', '', regex=True)

# Ensure file2's Base_ID is consistent
file2['Base_ID'] = file2['image_name'].str.replace(r'\.png\.png|\.png$', '', regex=True)

# Debugging check before merging
print("Unique Base_IDs in file1 (first 10):\n", file1['Base_ID'].unique()[:10])
print("Unique Base_IDs in file2 (first 10):\n", file2['Base_ID'].unique()[:10])

# Identify mismatches before merging
missing_in_file2 = file1[~file1['Base_ID'].isin(file2['Base_ID'])]
print("IDs in file1 that have no match in file2:\n", missing_in_file2[['file', 'Base_ID']].head())

# Select required columns from file2
file2_selected = file2[['Base_ID', 'assessment-5']].drop_duplicates(subset=['Base_ID'])

# Rename column for merging
file2_selected.rename(columns={'assessment-5': 'ID/OOD LLM'}, inplace=True)

# Drop existing 'ID/OOD LLM' column in file1 if it exists
if 'id/ood llm' in file1.columns:
   file1.drop(columns=['id/ood llm'], inplace=True)

# Merge the files on the extracted Base_ID
file1 = file1.merge(file2_selected, on='Base_ID', how='left')

# Fill NaN values with "No Match"
file1['ID/OOD LLM'] = file1['ID/OOD LLM'].fillna('No Match')

# Drop the temporary Base_ID column
file1.drop(columns=['Base_ID'], inplace=True)

# Remove any 'Unnamed' columns that might have been created
file1 = file1.loc[:, ~file1.columns.str.contains('^Unnamed')]

# Save the merged file
merged_file_path = 'data2.csv'  # Update with the desired output path
file1.to_csv(merged_file_path, index=False)

print("Merging complete. Output saved to", merged_file_path)
