import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_results(dataset_name, arch_name):
    result_dir = os.path.join("results", arch_name)
    if dataset_name == "ALL":
        datasets = sorted(set(f.split("_")[0] for f in os.listdir(result_dir) if f.endswith(".json")))
        return {d: load_results(d, arch_name) for d in datasets}
    else:
        files = [f for f in os.listdir(result_dir) if f.startswith(dataset_name) and f.endswith(".json")]
        return [json.load(open(os.path.join(result_dir, f), "r")) for f in files]

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


def _plot_dataset(dataset_name, all_metrics, metric, arch_name):
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

