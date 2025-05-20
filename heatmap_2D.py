import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_results(base_dir="results"):
    records = []
    for clf in os.listdir(base_dir):
        clf_dir = os.path.join(base_dir, clf)
        if not os.path.isdir(clf_dir):
            continue
        for file in os.listdir(clf_dir):
            if not file.endswith(".json"):
                continue
            filepath = os.path.join(clf_dir, file)
            with open(filepath) as f:
                data = json.load(f)
            dataset = data["dataset_stats"]["name"]
            val_acc = data.get("val_accuracy", [])
            # filters out None values
            val_acc = [v for v in val_acc if isinstance(v, (int, float)) and v is not None]

            # skips if the list is now empty
            if not val_acc:
                continue

            max_val = max(val_acc)
            avg_val = sum(val_acc) / len(val_acc)

            record = {
                "dataset": dataset,
                "classifier": clf,
                "max_val_accuracy": max_val,
                "avg_val_accuracy": avg_val
            }

            # adds each hyperparameter as a separate key
            for k, v in data.get("hyperparameters", {}).items():
                record[k] = v

            records.append(record)

    return pd.DataFrame(records)

def generate_heatmap(df, metric="max_val_accuracy"):
    pivot = df.groupby(["dataset", "classifier"])[metric].mean().unstack()
    plt.figure(figsize=(10, len(pivot) * 0.6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdPu")
    plt.title(f"Validation Accuracy on Convergence ({metric})")
    plt.xlabel("Classifier")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_results()
    generate_heatmap(df, metric="max_val_accuracy")