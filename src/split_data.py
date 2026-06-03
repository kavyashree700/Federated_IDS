import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading dataset...")

X = pd.read_csv("data/combined_features.csv")
y = pd.read_csv("data/combined_labels.csv")

# Combine features and labels
df = X.copy()
df["Label"] = y

print("Dataset Shape:", df.shape)

# Split into 4 equal parts
node1, temp = train_test_split(
    df,
    test_size=0.75,
    random_state=42
)

node2, temp = train_test_split(
    temp,
    test_size=0.6667,
    random_state=42
)

node3, node4 = train_test_split(
    temp,
    test_size=0.5,
    random_state=42
)

# Save datasets
node1.to_csv("data/camera_data.csv", index=False)
node2.to_csv("data/sensor_data.csv", index=False)
node3.to_csv("data/router_data.csv", index=False)
node4.to_csv("data/smartlock_data.csv", index=False)

print("Dataset split completed successfully!")

print("Camera Node:", node1.shape)
print("Sensor Node:", node2.shape)
print("Router Node:", node3.shape)
print("Smart Lock Node:", node4.shape)