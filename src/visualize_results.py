import pandas as pd
import matplotlib.pyplot as plt

print("Loading comparison results...")

df = pd.read_csv(
    "results/final_comparison.csv"
)

plt.figure(figsize=(6,5))

plt.bar(
    df["Approach"],
    df["Accuracy"]
)

plt.ylabel("Accuracy")
plt.xlabel("Approach")
plt.title("Centralized vs Federated IDS")

plt.tight_layout()

plt.savefig(
    "results/final_comparison_graph.png"
)

plt.show()

print("Graph Saved Successfully!")