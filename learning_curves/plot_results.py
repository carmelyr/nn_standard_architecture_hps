import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# this method returns the last non-None value in a list or np.nan if none found
def last_valid(values):
    return next((v for v in reversed(values) if v is not None and not np.isnan(v)), np.nan)


"""def load_results(dataset_name, arch_name):
    result_dir = os.path.join("results", arch_name)
    if dataset_name == "ALL":
        datasets = sorted(set(f.split("_")[0] for f in os.listdir(result_dir) if f.endswith(".json")))
        return {d: load_results(d, arch_name) for d in datasets}
    else:
        files = [f for f in os.listdir(result_dir) if f.startswith(dataset_name) and f.endswith(".json")]
        return [json.load(open(os.path.join(result_dir, f), "r")) for f in files]"""

def load_results(dataset_name, arch_name):
    result_dir = os.path.join("results", arch_name)

    if dataset_name == "ALL":
        # gets all dataset folders in the architecture directory
        dataset_names = sorted(d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d)))
        return {d: load_results(d, arch_name) for d in dataset_names}

    # all JSON result files for one dataset
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

# default architecture is FCNN, unless specified otherwise
def plot_learning_curves(dataset_name, metric="val_accuracy", arch_name="FCNN"):
    if dataset_name == "ALL":
        all_data = load_results("ALL", arch_name)
        for name, metrics in all_data.items():
            print(f"Plotting {name}...")
            _plot_dataset(name, metrics, metric, arch_name)
    else:
        all_metrics = load_results(dataset_name, arch_name)
        _plot_dataset(dataset_name, all_metrics, metric, arch_name)


"""def _plot_dataset(dataset_name, all_metrics, metric, arch_name):
    max_epochs = max(len(run[metric]) for run in all_metrics)
    metric_matrix = np.full((len(all_metrics), max_epochs), np.nan)

    for i, run in enumerate(all_metrics):
        metric_matrix[i, :len(run[metric])] = run[metric]

    mean_metric = np.nanmean(metric_matrix, axis=0)
    std_metric = np.nanstd(metric_matrix, axis=0)
    count_metric = np.sum(~np.isnan(metric_matrix), axis=0)
    last_epoch = np.max(np.where(count_metric >= 2)[0]) + 1
    plot_epochs = np.arange(1, last_epoch + 1)

    plt.figure(figsize=(10, 5))
    for run in all_metrics:
        epochs = np.arange(1, len(run[metric]) + 1)
        plt.plot(epochs, run[metric], color='blue', alpha=0.2, linewidth=1)

    plt.plot(plot_epochs, mean_metric[:last_epoch], color='blue', linewidth=2, label=f"Mean {metric} (n={len(all_metrics)})")
    plt.fill_between(plot_epochs, mean_metric[:last_epoch] - std_metric[:last_epoch], mean_metric[:last_epoch] + std_metric[:last_epoch], color='blue', alpha=0.2)

    plt.axvline(x=last_epoch, color='gray', linestyle='--', alpha=0.6)
    plt.text(last_epoch + 2, np.nanmin(mean_metric[:last_epoch]) + 0.02, f"Shaded area ends (n ≥ 2)\nEpoch {last_epoch}", color='gray', fontsize=9)

    plt.title(f"Validation Accuracy Over Epochs\n{dataset_name} ({arch_name})")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plot_dir = os.path.join(os.path.dirname(__file__), "plots", arch_name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{metric}.png"), dpi=300)
    plt.close()"""

def _plot_dataset(dataset_name, all_metrics, metric, arch_name):
    grouped = defaultdict(list)
    for config_dir in os.listdir(os.path.join("results", arch_name, dataset_name)):
        if not config_dir.startswith("config_"):
            continue
        config_path = os.path.join("results", arch_name, dataset_name, config_dir)
        for f in os.listdir(config_path):
            if f.endswith(".json") and f.startswith(dataset_name):
                config_id = config_dir.replace("config_", "")
                with open(os.path.join(config_path, f), "r") as file:
                    grouped[config_id].append(json.load(file))

    # plots mean and std of the metric for each config
    plt.figure(figsize=(10, 5))
    for config_id, runs in grouped.items():
        max_epochs = max(len(run[metric]) for run in runs)
        matrix = np.full((len(runs), max_epochs), np.nan)
        for i, run in enumerate(runs):
            cleaned = [v if v is not None else np.nan for v in run[metric]]
            matrix[i, :len(cleaned)] = cleaned

        mean_curve = np.nanmean(matrix, axis=0)
        std_curve = np.nanstd(matrix, axis=0)
        epochs = np.arange(1, len(mean_curve) + 1)

        # plots each config's mean curve
        plt.plot(epochs, mean_curve, color='#D1A0B1', alpha=0.3, lw=0.8)
        plt.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, color='#C51B7D', alpha=0.2)

    # plots the global mean and std of the metric across all configs
    global_max_len = max(len(r[metric]) for runs in grouped.values() for r in runs)

    all_matrix = []
    for runs in grouped.values():
        for r in runs:
            padded = np.full(global_max_len, np.nan)
            cleaned = [v if v is not None else np.nan for v in r[metric]]
            padded[:len(cleaned)] = cleaned
            all_matrix.append(padded)
    all_matrix = np.vstack(all_matrix)

    global_mean = np.nanmean(all_matrix, axis=0)
    global_std = np.nanstd(all_matrix, axis=0)
    epochs = np.arange(1, len(global_mean) + 1)
    plt.plot(epochs, global_mean, color='#7A0177', linewidth=2, label="Global Mean")
    plt.fill_between(epochs, global_mean - global_std, global_mean + global_std, color="#9D75F1", alpha=0.2)

    custom_lines = [
        Line2D([0], [0], color="#D1A0B1", lw=2, alpha=0.6, label='Per-config Mean'),
        Patch(facecolor='#C51B7D', edgecolor='none', label='Per-config Std Dev'),
        Line2D([0], [0], color='#7A0177', lw=2, label='Global Mean'),
        Patch(facecolor="#9D75F1", edgecolor='none', alpha=0.2, label='Global Std Dev')
    ]

    plt.legend(handles=custom_lines, loc='lower right')

    plt.title(f"Validation Accuracy Over Epochs\n{dataset_name} ({arch_name})")
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True)

    # counts of configurations and runs
    num_configs = len(grouped)
    runs_per_config = np.mean([len(runs) for runs in grouped.values()])
    total_runs = sum(len(runs) for runs in grouped.values())

    text = f"Configurations: {num_configs} | Runs/config: {int(runs_per_config)} | Total runs: {total_runs}"
    plt.text(0.01, 0.01, text, transform=plt.gca().transAxes, fontsize=9, color='gray', ha='left', va='bottom')

    last_vals = []
    for runs in grouped.values():
        for r in runs:
            cleaned = [v if v is not None else np.nan for v in r[metric]]
            last = last_valid(cleaned)
            last_vals.append(last)

    max_last = np.nanmax(last_vals)
    mean_last = np.nanmean(last_vals)
    print(f"[{dataset_name} - {arch_name}] Max final {metric}: {max_last:.4f} | Mean final: {mean_last:.4f}")

    plt.tight_layout()

    plot_dir = os.path.join(os.path.dirname(__file__), "plots", arch_name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{metric}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True, help="Dataset name(s) or 'ALL'")
    parser.add_argument("--arch", nargs="+", required=True, help="Architecture name(s), e.g., FCNN CNN")
    parser.add_argument("--metric", type=str, default="val_accuracy", help="Metric to plot (default: val_accuracy)")
    args = parser.parse_args()

    for arch in args.arch:
        for dataset in args.dataset:
            print(f"Plotting: {dataset} | {arch} | {args.metric}")
            plot_learning_curves(dataset, args.metric, arch)


