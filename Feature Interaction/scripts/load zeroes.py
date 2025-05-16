import os
import pandas as pd

# File name
filename = "zeros6.txt"

# Check if file exists
if not os.path.exists(filename):
    raise FileNotFoundError(f"File '{filename}' not found in current directory.")

# Load using a flexible whitespace separator
df = pd.read_csv(filename, sep=r'\s+', header=None, names=['ImaginaryPart'])

# Drop any rows with non-numeric data (just in case)
df = df[pd.to_numeric(df['ImaginaryPart'], errors='coerce').notnull()]

# Convert to float explicitly
df['ImaginaryPart'] = df['ImaginaryPart'].astype(float)

# Reset index for cleanliness
df.reset_index(drop=True, inplace=True)

# Show summary
print(df.info())
print("\nFirst few zeros:")
print(df.head())

# Optional: Save clean version to disk
df.to_csv("zeros6_clean.csv", index=False)
df.to_parquet("zeros6_clean.parquet", index=False)  # faster for big data
