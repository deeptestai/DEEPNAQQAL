import pandas as pd
import re

# Load the first file
file1_path = 'TIGvalidity - SVHN_SURVEY.csv'  # Update with actual path
file1 = pd.read_csv(file1_path)

# Load the second file
file2_path = 'mergedsvhn_llama2_classifications.csv'  # Update with actual path
file2 = pd.read_csv(file2_path)

# Normalize column names to lowercase and remove whitespace
file1.columns = file1.columns.str.strip().str.lower()
file2.columns = file2.columns.str.strip().str.lower()

# Verify column names for debugging
print("Columns in file1:", file1.columns)
print("Columns in file2:", file2.columns)

# Ensure correct column name for extracting Base_ID from file1
id_col = 'id' if 'id' in file1.columns else file1.columns[0]  # Auto-detect

# Function to extract "modelname_number" from file1
def extract_base_id(id_str):
    parts = id_str.split('_')
    base = '_'.join(parts[:2])  # Extracts 'dj_0'
    return base.lower().strip()

# Apply function to extract Base_ID correctly in file1
file1['Base_ID'] = file1[id_col].astype(str).apply(extract_base_id)

# Extract Base_ID from file2 and remove file extensions
file2['Base_ID'] = file2['image_name'].str.replace(r'(\.png\.png|\.png)', '', regex=True)
file2['Base_ID'] = file2['Base_ID'].apply(lambda x: '_'.join(x.split('_')[:2])).str.lower().str.strip()

# Debugging: Check unique Base_IDs before merging
print("Unique Base_IDs in file1 (first 10):\n", file1['Base_ID'].unique()[:10])
print("Unique Base_IDs in file2 (first 10):\n", file2['Base_ID'].unique()[:10])

# Identify mismatches before merging
missing_in_file2 = file1[~file1['Base_ID'].isin(file2['Base_ID'])]

# Debugging step: print missing IDs
if 'ID' in file1.columns:
    print("IDs in file1 that have no match in file2:\n", missing_in_file2[['id', 'Base_ID']].head())
else:
    print("Base_IDs in file1 that have no match in file2:\n", missing_in_file2[['Base_ID']].head())

# Select required columns from file2
file2_selected = file2[['Base_ID', 'assessment-2']].drop_duplicates(subset=['Base_ID'])

# Rename column for merging
file2_selected.rename(columns={'assessment-2': 'ID/OOD LLM2'}, inplace=True)

# Drop existing 'ID/OOD LLM2' column in file1 if it exists
if 'id/ood llm2' in file1.columns:
    file1.drop(columns=['id/ood llm2'], inplace=True)

# Merge the files on Base_ID
file1 = file1.merge(file2_selected, on='Base_ID', how='left')

# Fill NaN values with "No Match"
file1['ID/OOD LLM2'] = file1['ID/OOD LLM2'].fillna('No Match')

# Drop the temporary Base_ID column
file1.drop(columns=['Base_ID'], inplace=True)

# Remove any 'Unnamed' columns
file1 = file1.loc[:, ~file1.columns.str.contains('^Unnamed')]

# Save the merged file
merged_file_path = 'TIGvalidity - SVHN_SURVEY.csv'  # Update with the desired output path
file1.to_csv(merged_file_path, index=False)

print("Merging complete. Output saved to", merged_file_path)
