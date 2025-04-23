import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the CSV file
csv_file = "validation_results_mnist_deepsvdd.csv"
df = pd.read_csv(csv_file)

# Define the threshold (e.g., radius from DeepSVDD)
radius = 0.42893412709236145  # Example threshold value

# Add a Predicted column based on the threshold
df['Predicted'] = df['score'].apply(lambda x: 'ID' if x <= radius else 'OOD')

# Initialize metrics dictionary
metrics = []

# Group by TOOL (e.g., mnist_dj) and compute metrics
for tool, group in df.groupby('TOOL'):
    y_true = group['ID/OOD']  # Ground truth
    y_pred = group['Predicted']  # Model predictions

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['ID', 'OOD'])
    tn, fp, fn, tp = cm.ravel()

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Save metrics for this tool
    metrics.append({
        'Tool': tool,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Accuracy': accuracy
    })

# Save metrics to a new CSV file
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("mnist_validation_metrics.csv", index=False)

print("Metrics saved to mnist_validation_metrics.csv")
