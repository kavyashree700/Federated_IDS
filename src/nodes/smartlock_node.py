import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

print("Loading Smart Lock Node Dataset...")

# Load dataset
df = pd.read_csv("data/smartlock_data.csv")

# Features and Labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Fix label numbering
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Smart Lock Node XGBoost Model...")

# Train model
model = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Smart Lock Node Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(
    model,
    "models/smartlock_model.pkl"
)

# Save result
results = pd.DataFrame({
    "Node": ["Smart Lock"],
    "Accuracy": [accuracy]
})

results.to_csv(
    "results/smartlock_result.csv",
    index=False
)

print("Smart Lock Model Saved Successfully!")