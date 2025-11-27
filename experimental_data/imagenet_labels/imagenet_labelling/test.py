import pandas as pd

# Load the ID/OOD column from 'ood_analysis_stocco_all_classes1.csv'
file1 = "ood_analysis_stocco_all_classes2.csv"
df1 = pd.read_csv(file1, usecols=["SAMPLE", "ID/OOD"])

# Load the survey file 'TIGvalidity - MNIST_SURVEY.csv'
file2 = "data1.csv"
df2 = pd.read_csv(file2)

# Ensure correct column name mapping
sample_col = "SAMPLE"  # Column name in df1
id_col = "file"  # Column name in df2

# Clean column names to avoid leading/trailing spaces
df1[sample_col] = df1[sample_col].astype(str).str.strip()
df2[id_col] = df2[id_col].astype(str).str.strip()

# Convert all values to lowercase for case-insensitive matching
df1[sample_col] = df1[sample_col].str.lower()
df2[id_col] = df2[id_col].str.lower()

# Extract only the first two segments from SAMPLE and ID for comparison
df1["SAMPLE_cleaned"] = df1[sample_col].str.replace(".npy", "", regex=False).str.strip().str.extract(r'(^[a-z]+_\d+)')[0]
df2["ID_cleaned"] = df2[id_col].str.extract(r'(^[a-z]+_\d+)')[0]

# Handle NaN values in df1 before mapping
df1["ID/OOD"] = df1["ID/OOD"].fillna("Unknown")

# Debugging: Print unique values again after cleaning
print("Unique SAMPLE values in df1 (cleaned):", df1["SAMPLE_cleaned"].unique()[:10])
print("Unique ID values in df2 (cleaned):", df2["ID_cleaned"].unique()[:10])

# Add the new column 'ID/OOD Selforacle 99999' and populate it where SAMPLE matches df2 ID
df2["ID/OOD Selforacle 99999"] = df2["ID_cleaned"].map(df1.set_index("SAMPLE_cleaned")["ID/OOD"])

# Check if mapping worked correctly
print("Number of mapped values:", df2["ID/OOD Selforacle 99999"].notna().sum())

# Save the modified dataframe to a new file
output_file = "data2.csv"
df2.to_csv(output_file, index=False)

print(f"File saved as {output_file}")
