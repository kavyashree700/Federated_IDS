import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

print("Loading Camera Node Dataset...")

# Load Camera Node Data
df = pd.read_csv("data/camera_data.csv")

# Split Features and Labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Camera Node XGBoost Model...")

# Train Local Model
model = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

print("Training Completed!")

# Predictions
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print(f"Camera Node Accuracy: {accuracy:.4f}")

results = pd.DataFrame({
    "Node": ["Camera"],
    "Accuracy": [accuracy]
})

results.to_csv(
    "results/camera_result.csv",
    index=False
)

joblib.dump(
    model,
    "models/camera_model.pkl"
)

print("Camera Model Saved Successfully!")