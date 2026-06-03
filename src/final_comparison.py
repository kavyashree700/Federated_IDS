import pandas as pd

print("Loading Federated Result...")

# Best centralized XGBoost accuracy
centralized_accuracy = 0.99725

# Federated accuracy
federated_accuracy = pd.read_csv(
    "results/federated_result.csv"
)["Global Accuracy"][0]

comparison = pd.DataFrame({
    "Approach": [
        "Centralized XGBoost",
        "Federated IDS"
    ],
    "Accuracy": [
        centralized_accuracy,
        federated_accuracy
    ]
})

print("\nFinal Comparison")
print("----------------------")
print(comparison)

comparison.to_csv(
    "results/final_comparison.csv",
    index=False
)

print("\nComparison Saved Successfully!")