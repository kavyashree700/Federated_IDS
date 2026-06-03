import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

print("Loading Sensor Node Dataset...")

df = pd.read_csv("data/sensor_data.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

# Fix label numbering
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Sensor Node XGBoost Model...")

model = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print(f"Sensor Node Accuracy: {accuracy:.4f}")

results = pd.DataFrame({
    "Node": ["Sensor"],
    "Accuracy": [accuracy]
})

results.to_csv(
    "results/sensor_result.csv",
    index=False
)

joblib.dump(
    model,
    "models/sensor_model.pkl"
)

print("Sensor Model Saved Successfully!")