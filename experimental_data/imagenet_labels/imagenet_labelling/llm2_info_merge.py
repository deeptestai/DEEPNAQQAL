import pandas as pd
import re
from difflib import get_close_matches

# Load the first file
file1_path = 'data2.csv'  # Update with actual path
file1 = pd.read_csv(file1_path)

# Load the second file
file2_path = 'merged_llama3_classifications.csv'  # Update with actual path
file2 = pd.read_csv(file2_path)

# Normalize column names
file1.columns = file1.columns.str.strip()
file2.columns = file2.columns.str.strip()

# Remove ".npy" extension from Base_ID in file1
file1['Base_ID'] = file1['file'].str.replace(r'\.npy$', '', regex=True).str.strip().str.lower()

# Extract only the core Base_ID from file2 (remove all suffixes)
file2['Base_ID'] = file2['Image_Name'].apply(
    lambda x: re.sub(r'(_occl_|_light_|_blackout_)?_e\d+_p\d+|_blackout|_light|_occl|\.png', '', x)
).str.strip().str.lower()

# Convert Base_IDs to string type
file1['Base_ID'] = file1['Base_ID'].astype(str)
file2['Base_ID'] = file2['Base_ID'].astype(str)

# Debugging: Print unique Base_IDs
print("File1 Base_IDs (first 10):\n", file1['Base_ID'].unique()[:10])
print("File2 Base_IDs (first 10):\n", file2['Base_ID'].unique()[:10])

# Identify mismatches before merging
missing_ids = set(file1['Base_ID']) - set(file2['Base_ID'])
print("\nBase_IDs in file1 that are NOT in file2:\n", missing_ids)

# Identify close matches for debugging
close_matches = [x for x in file2['Base_ID'].unique() if any(y in x for y in missing_ids)]
print("\nClose Matches in file2 that are not exactly equal:\n", close_matches)

# Apply fuzzy matching for better alignment
file2_base_ids = file2['Base_ID'].tolist()
file1['Base_ID'] = file1['Base_ID'].apply(lambda x: get_close_matches(x, file2_base_ids, n=1, cutoff=0.9)[0] if get_close_matches(x, file2_base_ids, n=1, cutoff=0.9) else x)

# Select required columns from file2
file2_selected = file2[['Base_ID', 'assessment-5']].drop_duplicates(subset=['Base_ID'])

# Rename column for merging
file2_selected.rename(columns={'assessment-5': 'ID/OOD LLM3'}, inplace=True)

# Drop existing 'ID/OOD LLM2' column in file1 if it exists
if 'ID/OOD LLM3' in file1.columns:
    file1.drop(columns=['ID/OOD LLM3'], inplace=True)

# Merge the files on the extracted Base_ID
file1 = file1.merge(file2_selected, on='Base_ID', how='left')

# Check how many rows still have NaN after merge
print(f"\nRows with 'No Match' before filling: {file1['ID/OOD LLM3'].isna().sum()}")

# Fill NaN values with "No Match"
file1['ID/OOD LLM3'] = file1['ID/OOD LLM3'].fillna('No Match')

# Drop the temporary Base_ID column **(only once)**
file1.drop(columns=['Base_ID'], inplace=True)

# Remove any 'Unnamed' columns that might have been created
file1 = file1.loc[:, ~file1.columns.str.contains('^Unnamed')]

# Save the merged file
merged_file_path = 'data2.csv'  # Update with the desired output path
file1.to_csv(merged_file_path, index=False)

print("Merging complete. Output saved to", merged_file_path)
