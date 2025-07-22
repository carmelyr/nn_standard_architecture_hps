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

# plots the learning curves for a specific dataset and architecture
def _plot_dataset(dataset_name, all_metrics, metric, arch_name, smooth_window=5):
    from matplotlib.cm import get_cmap

    base_path = os.path.join("results", arch_name, dataset_name)
    grouped = defaultdict(list)

    # groups metrics by configuration ID
    for config_dir in os.listdir(base_path):
        if not config_dir.startswith("config_"):
            continue
        config_path = os.path.join(base_path, config_dir)
        for f in os.listdir(config_path):
            if f.endswith(".json") and f.startswith(dataset_name):
                config_id = config_dir.replace("config_", "")
                with open(os.path.join(config_path, f), "r") as file:
                    grouped[config_id].append(json.load(file))

    os.makedirs(os.path.join("learning_curves", "plots", arch_name, dataset_name), exist_ok=True)

    plt.figure(figsize=(12, 6))
    epochs = None
    final_vals = []

    for config_id, runs in grouped.items():
        seed_curves = []

        for run in runs:
            if "fold_logs" in run:
                for fold in run["fold_logs"]:
                    cleaned = [v if v is not None else np.nan for v in fold.get(metric, [])]
                    seed_curves.append(cleaned)
            else:
                cleaned = [v if v is not None else np.nan for v in run.get(metric, [])]
                seed_curves.append(cleaned)

        # pads all curves to same length
        max_len = max(len(c) for c in seed_curves)
        padded = np.full((len(seed_curves), max_len), np.nan)
        for i, curve in enumerate(seed_curves):
            padded[i, :len(curve)] = curve

        # average across folds + seeds
        config_mean = np.nanmean(padded, axis=0)
        config_mean_smoothed = smooth_curve(config_mean, window=smooth_window)

        epochs = np.arange(1, len(config_mean_smoothed) + 1)

        plt.plot(epochs, config_mean_smoothed, alpha=0.4, linewidth=1)

        final_vals.append(last_valid(config_mean_smoothed))

    plt.title(f"{metric.replace('val_', 'Validation ').title()} Over Epochs\n{dataset_name} ({arch_name}) - All Configs")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)

    text = f"{len(grouped)} configs × {len(runs)} seeds × {len(seed_curves) // len(runs)} folds"
    plt.text(0.01, 0.01, text, transform=plt.gca().transAxes, fontsize=9, color='gray', ha='left', va='bottom')

    filename = f"{dataset_name}_{metric}_{arch_name}.png"
    save_path = os.path.join("learning_curves", "plots", arch_name, dataset_name, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[{dataset_name} - {arch_name}] Final {metric}: max={np.nanmax(final_vals):.4f}, mean={np.nanmean(final_vals):.4f}")

    # ---- Representative curves ---- #
    print(f"Plotting representative curves for {dataset_name}...")

    # stores representative curves for each configuration
    config_curves = []

    for config_id, runs in grouped.items():
        seed_curves = []

        for run in runs:
            if "fold_logs" in run:
                for fold in run["fold_logs"]:
                    cleaned = [v if v is not None else np.nan for v in fold.get(metric, [])]
                    seed_curves.append(cleaned)
            else:
                cleaned = [v if v is not None else np.nan for v in run.get(metric, [])]
                seed_curves.append(cleaned)

        if not seed_curves:
            continue

        max_len = max(len(c) for c in seed_curves)
        padded = np.full((len(seed_curves), max_len), np.nan)
        for i, curve in enumerate(seed_curves):
            padded[i, :len(curve)] = curve

        config_mean = np.nanmean(padded, axis=0)
        config_mean_smoothed = smooth_curve(config_mean, window=smooth_window)

        def best_loss_score(curve, warmup=10):
            curve = np.array(curve)
            if np.all(np.isnan(curve)):
                return np.inf
            return np.nanmin(curve[warmup:])

        if metric.endswith("loss"):
            final_val = best_loss_score(config_mean_smoothed)
        else:
            final_val = last_valid(config_mean_smoothed)

        config_curves.append((config_id, final_val, config_mean_smoothed))

        # sorts curves by final value
        # if loss metric, lower is better
        ascending = metric.endswith("loss")
        config_curves.sort(key=lambda x: x[1], reverse=not ascending)

        n = len(config_curves)
        indices = {
            "Best": 0,
            "3rd Quartile": int(0.25 * (n - 1)),
            "Median": int(0.5 * (n - 1)),
            "1st Quartile": int(0.75 * (n - 1)),
            "Worst": n - 1
        }

    colors = {
        "Worst": "#d73027",
        "1st Quartile": "#fc8d59",
        "Median": "#fee090",
        "3rd Quartile": "#91bfdb",
        "Best": "#4575b4"
    }

    plt.figure(figsize=(12, 6))
    for label, idx in indices.items():
        config_id, _, curve = config_curves[idx]
        epochs = np.arange(1, len(curve) + 1)
        plt.plot(epochs, curve, label=f"{label} (config_{config_id})", color=colors[label], linewidth=2)

    plt.title(f"Representative Configs ({dataset_name} - {arch_name})\n{metric.replace('val_', 'Validation ').title()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)
    plt.legend()

    rep_dir = os.path.join("learning_curves", "representative_curves", arch_name, dataset_name)
    os.makedirs(rep_dir, exist_ok=True)
    rep_filename = f"{dataset_name}_{metric}_{arch_name}_representative.png"
    rep_path = os.path.join(rep_dir, rep_filename)
    plt.tight_layout()
    plt.savefig(rep_path, dpi=300)
    plt.close()
    print(f"Representative curves saved to {rep_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True, help="Dataset name(s) or 'ALL'")
    parser.add_argument("--arch", nargs="+", required=True, help="Architecture name(s), e.g., FCNN CNN")
    parser.add_argument("--metric", type=str, default="val_loss", help="Metric to plot (e.g., val_accuracy, val_loss)")
    args = parser.parse_args()

    for arch in args.arch:
        for dataset in args.dataset:
            print(f"Plotting: {dataset} | {arch} | {args.metric}")
            plot_learning_curves(dataset, args.metric, arch)