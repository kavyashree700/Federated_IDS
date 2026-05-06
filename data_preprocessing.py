import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:\Users\admin\OneDrive\Desktop\federated_ids\data\Friday.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Clean column names FIRST
df.columns = df.columns.str.strip()

print("Original Shape:", df.shape)

# -------------------------------
# 1. Remove duplicate rows
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# 2. Handle missing values
# -------------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# -------------------------------
# 3. Encode labels
# -------------------------------
print("Unique Labels:", df['Label'].unique())

df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# -------------------------------
# 4. Feature / Target split
# -------------------------------
X = df.drop('Label', axis=1)
y = df['Label']

# -------------------------------
# 5. Save cleaned data
# -------------------------------
X.to_csv("../data/clean_features.csv", index=False)
y.to_csv("../data/clean_labels.csv", index=False)

print("Cleaned Shape:", df.shape)
print("Data preprocessing completed successfully!")