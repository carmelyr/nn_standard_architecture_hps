# Heatmap for all models and datasets
# This script loads results from a directory structure and generates a heatmap based on validation accuracy metric.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from json import JSONDecodeError

# this function finds the last valid value in a list
def last_valid(values):
    return next((v for v in reversed(values) if v is not None and not np.isnan(v)), np.nan)     # skips NaNs

# this function applies a centered rolling mean to smooth the metric curve
def smooth_curve(values, window=5):
    series = pd.Series(values)
    return series.rolling(window, min_periods=1, center=True).mean().to_numpy()

# this function loads the results from the specified directory
def load_results(base_dir="results", metric="val_accuracy", smooth_window=5):

    records = []

    for clf in os.listdir(base_dir):
        clf_dir = os.path.join(base_dir, clf)
        if not os.path.isdir(clf_dir):
            continue

        for dataset_name in os.listdir(clf_dir):
            dataset_dir = os.path.join(clf_dir, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue

            for config in os.listdir(dataset_dir):
                config_dir = os.path.join(dataset_dir, config)
                if not os.path.isdir(config_dir):
                    continue

                seed_curves = []
                for file in os.listdir(config_dir):
                    if not file.endswith(".json") or not file.startswith(dataset_name):
                        continue

                    file_path = os.path.join(config_dir, file)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                    except (JSONDecodeError, ValueError) as e:
                        print(f"Skipping invalid JSON file: {file_path} ({e})")
                        continue

                    if "fold_logs" in data:
                        for fold in data["fold_logs"]:
                            val_acc = [v if v is not None else np.nan for v in fold.get(metric, [])]
                            seed_curves.append(val_acc)
                    else:
                        val_acc = [v if v is not None else np.nan for v in data.get(metric, [])]
                        seed_curves.append(val_acc)

                if not seed_curves:
                    continue

                max_len = max(len(c) for c in seed_curves)
                padded = np.full((len(seed_curves), max_len), np.nan)
                for i, curve in enumerate(seed_curves):
                    padded[i, :len(curve)] = curve

                config_mean = np.nanmean(padded, axis=0)        # single averaged learning curve representing all seeds
                config_mean_smoothed = smooth_curve(config_mean, window=smooth_window)  # 5-point rolling average to reduce noise
                final_val = last_valid(config_mean_smoothed)  # gets final validation accuracy (last valid value from the smoothed curve)

                record = {
                    "dataset": dataset_name,
                    "classifier": clf,
                    "config_id": config,
                    "max_val_accuracy": final_val,            # max across configs in heatmap
                    "avg_val_accuracy": final_val             # average across configs in heatmap
                }

                records.append(record)

    return pd.DataFrame(records)

# this function generates a heatmap from the results DataFrame
def generate_heatmap(df, metric="max_val_accuracy"):
    pivot = df.groupby(["dataset", "classifier"])[metric].max().unstack()       # for max val accuracy
    #pivot = df.groupby(["dataset", "classifier"])[metric].mean().unstack()     # for average val accuracy

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
    plt.xlabel("Neural Network Architecture")
    plt.ylabel("Dataset")
    plt.tight_layout()
    
    os.makedirs("heatmap_2D_img", exist_ok=True)
    output_path = os.path.join("heatmap_2D_img", f"validation_accuracy_heatmap_{metric}.pdf")
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved heatmap to: {output_path}")

if __name__ == "__main__":
    df = load_results(metric="val_accuracy")
    generate_heatmap(df, metric="max_val_accuracy")
