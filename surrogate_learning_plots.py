# This script generates combined learning curves for surrogate models
# It reads JSON files containing training metrics and plots the results

import os
import json
import matplotlib.pyplot as plt

regressors = ["XGBoost", "RandomForest", "LinearRegression"]
metrics = ["r2", "rmse", "mae"]
titles = ["R² Score", "RMSE", "MAE"]
colors = {"XGBoost": "FireBrick", "RandomForest": "DarkGreen", "LinearRegression": "DarkBlue"}

# location of learning_curve_data/<Model>/<Regressor>_<Dataset>.json
base_dir = "learning_curve_data"

# loops through each model directory
for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_path):
        continue

    # gets all datasets for the current model from the JSON files
    datasets = set()
    for fname in os.listdir(model_path):
        if fname.endswith(".json"):
            parts = fname.split("_")
            if len(parts) > 1:
                datasets.add("_".join(parts[1:]).replace(".json", ""))

    # plots a figure for each dataset
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
                axs[i].errorbar(train_sizes, means, yerr=stds, label=regressor, marker='o', capsize=4, color=colors[regressor])
                axs[i].set_title(f"{titles[i]} vs Training Size — {dataset}")
                axs[i].set_xlabel("Number of Training Examples")
                axs[i].set_ylabel(titles[i])
                axs[i].grid(True)

        for ax in axs:
            ax.legend()
        plt.tight_layout()
        out_dir = os.path.join("combined_learning_curves", model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{model_name}_{dataset}_combined_learning_curve.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f" Saved combined learning curve to: {out_path}")
