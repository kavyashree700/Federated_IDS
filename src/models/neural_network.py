import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# --------------------------------
# Load Dataset
# --------------------------------
print("Loading dataset...")

X = pd.read_csv("data/combined_features.csv")
y = pd.read_csv("data/combined_labels.csv")

# Convert labels into array
y = y.values.ravel()

print("Dataset loaded successfully!")

# --------------------------------
# Train Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train test split completed!")

# --------------------------------
# Feature Scaling
# --------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed!")

# --------------------------------
# Train Neural Network Model
# --------------------------------
print("Training Neural Network model...")

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

print("Neural Network training completed!")

# --------------------------------
# Predictions
# --------------------------------
y_pred = model.predict(X_test)

# --------------------------------
# Evaluation Metrics
# --------------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

# --------------------------------
# Print Results
# --------------------------------
print("\nNeural Network Results")
print("----------------------------")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# --------------------------------
# Classification Report
# --------------------------------
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# --------------------------------
# Save Results
# --------------------------------
results = pd.DataFrame({
    "Algorithm": ["Neural Network"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

results.to_csv(
    "results/neural_network_results.csv",
    index=False
)

print("\nResults saved successfully!")