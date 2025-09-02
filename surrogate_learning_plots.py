# This script generates combined learning curves for surrogate models
# it reads JSON files containing training metrics and plots the results

import os
import json
import matplotlib.pyplot as plt
import math

regressors = ["XGBoost", "RandomForest", "LinearRegression"]
metrics = ["r2", "rmse", "mae"]
titles = ["R² Score", "RMSE", "MAE"]
colors = {"XGBoost": "FireBrick", "RandomForest": "DarkGreen", "LinearRegression": "DarkBlue"}
markers = {"XGBoost": "o", "RandomForest": "s", "LinearRegression": "^"}

base_dir = "learning_curve_data"

# ---------- PASS 1: computes global ranges from means only ----------
global_ranges = {}
for metric in metrics:
    global_ranges[metric] = []

for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_path):
        continue
    for fname in os.listdir(model_path):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(model_path, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            for metric in metrics:
                means = data.get(f"{metric}_mean", None)
                if means is None:
                    continue
                for v in means:
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        global_ranges[metric].append(float(v))
        except Exception:
            pass

# calculates global limits for each metric
global_limits = {}
for metric in metrics:
    if global_ranges[metric]:
        min_val = min(global_ranges[metric])
        max_val = max(global_ranges[metric])
        
        if metric == "r2":
            # R² cannot exceed 1.0; allows a tiny headroom for errorbar caps
            min_val = max(-1.0, min_val)
            max_val = min(1.0, max_val)
        
        # pads & clamps to a sensible range
        rng = max(max_val - min_val, 0.05)  # avoids zero range
        pad = 0.02 * rng
        
        if metric == "r2":
            global_limits[metric] = (max(-1.0, min_val - pad), min(1.02, max_val + pad))
        else:
            # for RMSE and MAE, ensures that lower bound is not negative
            global_limits[metric] = (max(0.0, min_val - pad), max_val + pad)
    else:
        if metric == "r2":
            global_limits[metric] = (0.0, 1.0)
        else:
            global_limits[metric] = (0.0, 1.0)
# ---------------------------------------------------------------------


# ---------- PASS 2: plotting ----------
for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_path):
        continue

    # collects datasets present for this model
    datasets = set()
    for fname in os.listdir(model_path):
        if fname.endswith(".json"):
            parts = fname.split("_")
            if len(parts) > 1:
                datasets.add("_".join(parts[1:]).replace(".json", ""))

    for dataset in sorted(datasets):
        fig, axs = plt.subplots(3, 1, figsize=(7, 12))

        for regressor in regressors:
            json_file = os.path.join(model_path, f"{regressor}_{dataset}.json")
            if not os.path.exists(json_file):
                continue

            with open(json_file, "r") as f:
                data = json.load(f)

            train_sizes = data["train_sizes"]

            for i, metric in enumerate(metrics):
                means = data[f"{metric}_mean"]
                stds = data[f"{metric}_std"]
                
                # calculates average standard deviation for the legend
                avg_std = sum(stds) / len(stds)
                
                axs[i].errorbar(train_sizes, means, yerr=stds,label=f"{regressor} (σ={avg_std:.3f})", marker=markers[regressor], capsize=4, color=colors[regressor])
                axs[i].set_title(f"{titles[i]} vs Training Size — {model_name} — {dataset}")
                axs[i].set_xlabel("Number of Training Examples")
                axs[i].set_ylabel(titles[i])
                axs[i].set_xticks([20, 40, 60, 80])
                axs[i].grid(True)

                # enforces global scale for all metrics
                axs[i].set_ylim(global_limits[metric][0], global_limits[metric][1])

        for ax in axs:
            ax.legend()
        plt.tight_layout()
        out_dir = os.path.join("combined_learning_curves", model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{model_name}_{dataset}_combined_learning_curve.pdf")
        plt.savefig(out_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f" Saved combined learning curve to: {out_path}")
