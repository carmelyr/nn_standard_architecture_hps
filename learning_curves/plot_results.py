import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# returns the last valid (non-NaN) value in a list
def last_valid(values):
    return next((v for v in reversed(values) if v is not None and not np.isnan(v)), np.nan)

def load_results(dataset_name, arch_name):
    result_dir = os.path.join("results", arch_name)

    if dataset_name == "ALL":
        dataset_names = sorted(d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d)))
        return {d: load_results(d, arch_name) for d in dataset_names}

    dataset_dir = os.path.join(result_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        return []

    all_metrics = []
    for config_dir in sorted(os.listdir(dataset_dir)):
        config_path = os.path.join(dataset_dir, config_dir)
        if not os.path.isdir(config_path):
            continue
        for filename in os.listdir(config_path):
            if filename.endswith(".json") and filename.startswith(dataset_name):
                file_path = os.path.join(config_path, filename)
                with open(file_path, "r") as f:
                    all_metrics.append(json.load(f))
    return all_metrics

def plot_learning_curves(dataset_name, metric="val_accuracy", arch_name="FCNN"):
    if dataset_name == "ALL":
        all_data = load_results("ALL", arch_name)
        for name, metrics in all_data.items():
            print(f"Plotting {name}...")
            _plot_dataset(name, metrics, metric, arch_name)
    else:
        all_metrics = load_results(dataset_name, arch_name)
        _plot_dataset(dataset_name, all_metrics, metric, arch_name)

# applies a centered rolling mean to smooth the metric curve
def smooth_curve(values, window=5):
    series = pd.Series(values)
    return series.rolling(window, min_periods=1, center=True).mean().to_numpy()

def _plot_dataset(dataset_name, all_metrics, metric, arch_name, smooth_window=5):
    # groups metrics by configuration ID
    grouped = defaultdict(list)
    base_path = os.path.join("results", arch_name, dataset_name)
    for config_dir in os.listdir(base_path):
        if not config_dir.startswith("config_"):
            continue
        config_path = os.path.join(base_path, config_dir)
        for f in os.listdir(config_path):
            if f.endswith(".json") and f.startswith(dataset_name):
                config_id = config_dir.replace("config_", "")
                with open(os.path.join(config_path, f), "r") as file:
                    grouped[config_id].append(json.load(file))

    plt.figure(figsize=(10, 5))

    # loads all curves for the specified metric
    all_curves = []
    for runs in grouped.values():
        for r in runs:
            if "fold_logs" in r:
                for fold in r["fold_logs"]:
                    cleaned = [v if v is not None else np.nan for v in fold[metric]]    # replaces None with NaN
                    all_curves.append(cleaned)
            else:
                cleaned = [v if v is not None else np.nan for v in r[metric]]           # replaces None with NaN
                all_curves.append(cleaned)

    # pads the curves to the same length for plotting
    max_len = max(len(c) for c in all_curves)
    padded_matrix = np.full((len(all_curves), max_len), np.nan)
    for i, curve in enumerate(all_curves):
        padded_matrix[i, :len(curve)] = curve

    # smooths the curves using a rolling mean
    smoothed_matrix = np.array([smooth_curve(row, window=smooth_window) for row in padded_matrix])

    # aggregates the smoothed curves: calculates the global mean and standard deviation
    global_mean = np.nanmean(smoothed_matrix, axis=0)
    global_std = np.nanstd(smoothed_matrix, axis=0)
    epochs = np.arange(1, max_len + 1)

    plt.plot(epochs, global_mean, color="#FF01B3", linewidth=2, label="Global Mean")
    lower = np.clip(global_mean - global_std, 0.0, 1.0)
    upper = np.clip(global_mean + global_std, 0.0, 1.0)
    plt.fill_between(epochs, lower, upper, color="#77BDFF", alpha=0.2)

    custom_lines = [Line2D([0], [0], color='#FF01B3', lw=2, label='Global Mean'), Patch(facecolor="#77BDFF", edgecolor='none', alpha=0.2, label='Global Std Dev')]

    plt.legend(handles=custom_lines, loc='lower right')
    plt.title(f"{metric.replace('val_', 'Validation ').title()} Over Epochs\n{dataset_name} ({arch_name})")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)

    total_runs = len(all_curves)
    num_configs = len(grouped)
    first_config_runs = next(iter(grouped.values()))
    first_run = first_config_runs[0]
    num_seeds = len(first_config_runs)
    num_folds = len(first_run["fold_logs"])

    total_runs = num_configs * num_seeds * num_folds        # calculates total runs: configs × seeds × folds
    text = f"Total runs: {num_configs} configs × {num_seeds} seeds × {num_folds} folds = {total_runs}"
    plt.text(0.01, 0.01, text, transform=plt.gca().transAxes, fontsize=9, color='gray', ha='left', va='bottom')

    last_vals = [last_valid(row) for row in padded_matrix]
    max_last = np.nanmax(last_vals)
    mean_last = np.nanmean(last_vals)
    print(f"[{dataset_name} - {arch_name}] Max final {metric}: {max_last:.4f} | Mean final: {mean_last:.4f}")

    plt.tight_layout()
    metric_dirname = f"{metric}_curves"
    plot_dir = os.path.join(os.path.dirname(__file__), "plots", arch_name, metric_dirname)
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"{dataset_name}_{metric}_{arch_name}.png"
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True, help="Dataset name(s) or 'ALL'")
    parser.add_argument("--arch", nargs="+", required=True, help="Architecture name(s), e.g., FCNN CNN")
    parser.add_argument("--metric", type=str, default="val_accuracy", help="Metric to plot (e.g., val_accuracy, val_loss)")
    args = parser.parse_args()

    for arch in args.arch:
        for dataset in args.dataset:
            print(f"Plotting: {dataset} | {arch} | {args.metric}")
            plot_learning_curves(dataset, args.metric, arch)