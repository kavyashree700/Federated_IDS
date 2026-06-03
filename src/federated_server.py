import pandas as pd

print("Loading Local Node Results...")

# Load node accuracies
camera = pd.read_csv("results/camera_result.csv")
sensor = pd.read_csv("results/sensor_result.csv")
router = pd.read_csv("results/router_result.csv")
smartlock = pd.read_csv("results/smartlock_result.csv")

# Extract accuracy values
camera_acc = camera["Accuracy"][0]
sensor_acc = sensor["Accuracy"][0]
router_acc = router["Accuracy"][0]
smartlock_acc = smartlock["Accuracy"][0]

print("\nNode Accuracies")
print("-----------------------")
print(f"Camera Node     : {camera_acc:.4f}")
print(f"Sensor Node     : {sensor_acc:.4f}")
print(f"Router Node     : {router_acc:.4f}")
print(f"Smart Lock Node : {smartlock_acc:.4f}")

# Simulated FedAvg
global_accuracy = (
    camera_acc +
    sensor_acc +
    router_acc +
    smartlock_acc
) / 4

print("\nFederated Learning Result")
print("-----------------------")
print(f"Global Accuracy : {global_accuracy:.4f}")

# Save result
result = pd.DataFrame({
    "Global Accuracy": [global_accuracy]
})

result.to_csv(
    "results/federated_result.csv",
    index=False
)

print("\nFederated Result Saved Successfully!")