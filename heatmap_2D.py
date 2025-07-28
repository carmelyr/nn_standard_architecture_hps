# Heatmap for all models and datasets
# This script loads results from a directory structure and generates a heatmap

import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from json import JSONDecodeError

def load_results(base_dir="results"):
    records = []
    for clf in os.listdir(base_dir):
        clf_dir = os.path.join(base_dir, clf)
        if not os.path.isdir(clf_dir):
            continue
            
        # dataset folders inside classifier folder
        for dataset_name in os.listdir(clf_dir):
            dataset_dir = os.path.join(clf_dir, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue
                
            # config folders inside dataset folder
            for config in os.listdir(dataset_dir):
                config_dir = os.path.join(dataset_dir, config)
                if not os.path.isdir(config_dir):
                    continue
                    
                # JSON files in the config directory
                for file in os.listdir(config_dir):
                    if not file.endswith(".json"):
                        continue
                        
                    filepath = os.path.join(config_dir, file)
                    try:
                        with open(filepath) as f:
                            data = json.load(f)
                    except (JSONDecodeError, ValueError) as e:
                        print(f"Skipping invalid JSON file: {filepath} ({e})")
                        continue
                        
                    dataset = data["dataset_stats"]["name"]
                    val_acc = data.get("val_accuracy", [])
                    val_acc = [v for v in val_acc if isinstance(v, (int, float)) and v is not None]

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

                    for k, v in data.get("hyperparameters", {}).items():
                        record[k] = v

                    records.append(record)

    return pd.DataFrame(records)

def generate_heatmap(df, metric="max_val_accuracy"):
    pivot = df.groupby(["dataset", "classifier"])[metric].max().unstack()
    #pivot = df.groupby(["dataset", "classifier"])[metric].mean().unstack()

    plt.figure(figsize=(12, len(pivot) * 0.3))
    
    ax = sns.heatmap(pivot, annot=False, fmt=".2f", cmap="RdPu", vmin=0, vmax=1)
    
    for i, row_idx in enumerate(pivot.index):
        row_data = pivot.loc[row_idx]
        max_val = row_data.max()
        
        for j, col_idx in enumerate(pivot.columns):
            value = pivot.loc[row_idx, col_idx]
            if pd.isna(value):
                continue
                
            # checks if the value is highest in the row
            is_max = abs(value - max_val) < 1e-10
            
            text = f"{value:.2f}"
            if is_max:
                ax.text(j + 0.5, i + 0.5, text, 
                       horizontalalignment='center', verticalalignment='center',
                       fontweight='bold', fontsize=10, color='white')
            else:
                ax.text(j + 0.5, i + 0.5, text,
                       horizontalalignment='center', verticalalignment='center',
                       fontweight='normal', fontsize=10, color='white')
    
    plt.title(f"Validation Accuracy Heatmap ({metric})", fontweight='bold', fontsize=12)
    plt.xlabel("Classifier")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_results()
    generate_heatmap(df, metric="max_val_accuracy")