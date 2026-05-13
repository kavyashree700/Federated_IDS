import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# Load Results
# --------------------------------
print("Loading model results...")

rf = pd.read_csv("results/random_forest_results.csv")

xgb = pd.read_csv("results/xgboost_results.csv")

nn = pd.read_csv("results/neural_network_results.csv")

print("Results loaded successfully!")

# --------------------------------
# Combine Results
# --------------------------------
comparison = pd.concat([rf, xgb, nn], ignore_index=True)

print("\nFinal Comparison Table:")
print(comparison)

# --------------------------------
# Save Final Comparison
# --------------------------------
comparison.to_csv(
    "results/final_comparison.csv",
    index=False
)

print("\nFinal comparison saved successfully!")

# --------------------------------
# Find Best Model
# --------------------------------
best_model = comparison.loc[
    comparison['Accuracy'].idxmax()
]

print("\nBest Model Selected:")
print(best_model)

# --------------------------------
# Plot Accuracy Graph
# --------------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Algorithm"],
    comparison["Accuracy"]
)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("IDS Algorithm Comparison")

plt.savefig(
    "results/comparison_graph.png"
)

print("\nComparison graph saved successfully!")

plt.show()