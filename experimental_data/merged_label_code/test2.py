import pandas as pd

# Load the ID/OOD column from 'ood_analysis_stocco_all_classes2.csv'
file1 = "ood_analysis_stocco_all_classes3.csv"
df1 = pd.read_csv(file1, usecols=["SAMPLE", "ID/OOD"])

# Load the survey file 'TIGvalidity - MNIST_SURVEY.csv'
file2 = "TIGvalidity - MNIST_SURVEY.csv"
df2 = pd.read_csv(file2)

# Ensure correct column name mapping
sample_col = "SAMPLE"  # Column name in df1
id_col = "ID"  # Column name in df2

# Clean column values to avoid leading/trailing spaces
df1[sample_col] = df1[sample_col].astype(str).str.strip()
df2[id_col] = df2[id_col].astype(str).str.strip()

# Convert values to lowercase for case-insensitive matching (column names remain unchanged)
df1[sample_col] = df1[sample_col].str.lower()
df2[id_col] = df2[id_col].str.lower()

# Remove .npy from SAMPLE column in df1 and extract matching segments
df1["SAMPLE_cleaned"] = df1[sample_col].str.replace(".npy", "", regex=False).str.strip().str.extract(r'(^[a-z]+_\d+)')[0]

# Extract only the dj_XX part from df2["ID"]
df2["ID_extracted"] = df2[id_col].str.extract(r'(^[a-z]+_\d+)')[0]

# Debugging: Print unique extracted values
print("\nUnique SAMPLE_cleaned values in df1:", df1["SAMPLE_cleaned"].unique()[:10])
print("\nUnique extracted ID values in df2:", df2["ID_extracted"].unique()[:10])

# Check if "ID/OOD Selforacle 99" exists in df2 and drop it before adding a new one
if "ID/OOD Selforacle 99" in df2.columns:
    df2.drop(columns=["ID/OOD Selforacle 99"], inplace=True)

# Perform the mapping
df2["ID/OOD Selforacle 999"] = df2["ID_extracted"].map(df1.set_index("SAMPLE_cleaned")["ID/OOD"])

# Check if mapping worked correctly
print("\nNumber of mapped values after mapping:", df2["ID/OOD Selforacle 999"].notna().sum())

# Drop the "ID_extracted" column after mapping
df2.drop(columns=["ID_extracted"], inplace=True)

# Save the modified dataframe to a new file
output_file = "TIGvalidity - MNIST_SURVEY.csv"
df2.to_csv(output_file, index=False)

print(f"\nFile saved as {output_file}")
