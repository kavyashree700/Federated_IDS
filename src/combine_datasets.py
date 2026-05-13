import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# -------------------------------
# Load multiple datasets
# -------------------------------
files = [
    "data/Monday.csv",
    "data/Tuesday.csv",
    "data/Friday.csv",
    "data/Friday1.csv",
    "data/Wednesday.csv",
    "data/Thursday1.csv",
    "data/Thursday2.csv"
]

df_list = []

for file in files:
    print(f"Loading {file}...")
    temp_df = pd.read_csv(file, encoding='latin1')
    temp_df.columns = temp_df.columns.str.strip()
    df_list.append(temp_df)

# -------------------------------
# Combine datasets
# -------------------------------
df = pd.concat(df_list, ignore_index=True)

print("Combined Shape:", df.shape)

# -------------------------------
# Remove duplicates
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# Handle missing values
# -------------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# -------------------------------
# Reduce size (IMPORTANT)
# -------------------------------
df = df.sample(20000, random_state=42)

# -------------------------------
# Convert labels (Binary)
# -------------------------------
print("Labels before encoding:", df['Label'].unique())



le = LabelEncoder()

df['Label'] = le.fit_transform(df['Label'])


print("\nEncoded label Counts:")
print(df['Label'].value_counts())
# -------------------------------
# Split features & labels
# -------------------------------
X = df.drop('Label', axis=1)
y = df['Label']

# -------------------------------
# Save output
# -------------------------------
X.to_csv("data/combined_features.csv", index=False)
y.to_csv("data/combined_labels.csv", index=False)

print("Preprocessing completed successfully!")
print("Final Shape:", df.shape)