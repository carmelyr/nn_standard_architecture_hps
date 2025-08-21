import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sktime.datasets import load_from_tsfile
from scipy.io import arff

# returns the last valid (non-NaN) value in a list
def last_valid(values):
    return next((v for v in reversed(values) if v is not None and not np.isnan(v)), np.nan)

def get_majority_class_baseline(dataset_name):
    """Calculate the majority class baseline accuracy for a dataset."""
    # Define the dataset paths based on the patterns seen in the codebase
    dataset_paths = {
        "classification_ozone": "datasets/classification_ozone/y_train.csv",
        # Most other datasets follow the pattern: datasets/{name}/{name}_TRAIN.{ext}
    }
    
    # First try the specific path if defined
    if dataset_name in dataset_paths:
        try:
            y_train = pd.read_csv(dataset_paths[dataset_name]).values.squeeze().astype(int)
            unique, counts = np.unique(y_train, return_counts=True)
            return np.max(counts) / len(y_train)
        except:
            pass
    
    # Try different file extensions and patterns
    possible_paths = [
        f"datasets/{dataset_name}/{dataset_name}_TRAIN.txt",
        f"datasets/{dataset_name}/{dataset_name}_TRAIN.ts",
        f"datasets/{dataset_name}/{dataset_name}_TRAIN.arff",
    ]
    
    for path in possible_paths:
        if not os.path.exists(path):
            continue
            
        try:
            if path.endswith(".txt"):
                train_data = np.loadtxt(path)
                y_train = train_data[:, 0].astype(int)
            elif path.endswith(".ts"):
                X, y = load_from_tsfile(path)
                y_train = pd.factorize(y)[0]  # Convert to integer labels
            elif path.endswith(".arff"):
                data, meta = arff.loadarff(path)
                y_train = data[meta.names[-1]]  # Last column is usually the target
                if hasattr(y_train[0], 'decode'):  # Handle byte strings
                    y_train = [y.decode() if hasattr(y, 'decode') else y for y in y_train]
                y_train = pd.factorize(y_train)[0]
            
            # Calculate majority class proportion
            unique, counts = np.unique(y_train, return_counts=True)
            return np.max(counts) / len(y_train)
            
        except Exception as e:
            continue
    
    # If we can't load the dataset, return None (will skip baseline)
    print(f"Warning: Could not calculate baseline for {dataset_name}")
    return None

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
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        all_metrics.append(data)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Warning: Skipping corrupted file {file_path}: {e}")
                    continue
    return all_metrics

def get_dataset_max_accuracy(dataset_name, metric="val_accuracy"):
    """Get the maximum accuracy achieved by any model for this dataset."""
    max_acc = 0.0
    models = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
    
    for model in models:
        try:
            all_metrics = load_results(dataset_name, model)
            if not all_metrics:
                continue
                
            for fold_data in all_metrics:
                if metric in fold_data:
                    curve_max = np.nanmax(fold_data[metric])
                    if not np.isnan(curve_max):
                        max_acc = max(max_acc, curve_max)
        except:
            continue
    
    return max_acc if max_acc > 0 else 1.0

def get_dataset_max_epochs(dataset_name):
    """Get the maximum number of epochs across all models for this dataset."""
    max_epochs = 0
    models = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
    
    for model in models:
        try:
            all_metrics = load_results(dataset_name, model)
            if not all_metrics:
                continue
                
            for fold_data in all_metrics:
                # Check different possible metric keys to find epoch counts
                for key in ["val_accuracy", "val_loss", "train_accuracy", "train_loss"]:
                    if key in fold_data:
                        epochs = len(fold_data[key])
                        max_epochs = max(max_epochs, epochs)
                        break
                
                # Also check fold_logs if present
                if "fold_logs" in fold_data:
                    for fold in fold_data["fold_logs"]:
                        for key in ["val_accuracy", "val_loss", "train_accuracy", "train_loss"]:
                            if key in fold:
                                epochs = len(fold[key])
                                max_epochs = max(max_epochs, epochs)
                                break
        except:
            continue
    
    return max_epochs if max_epochs > 0 else 100  # fallback to 100 epochs

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

    def get_reach_epoch(curve, target, tolerance=1e-4):
        try:
            return next(i for i, v in enumerate(curve) if v >= target - tolerance)
        except StopIteration:
            return len(curve)

    def get_stability(curve, start_epoch=0):
        curve = np.array(curve[start_epoch:])
        diffs = np.diff(curve)
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) == 0:
            return 0.0
        return -np.mean(np.abs(diffs))  # negative = higher stability is better

    # Get maximum epochs for this dataset across all models
    dataset_max_epochs = get_dataset_max_epochs(dataset_name)

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
                try:
                    with open(os.path.join(config_path, f), "r") as file:
                        data = json.load(file)
                        grouped[config_id].append(data)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Warning: Skipping corrupted file {os.path.join(config_path, f)}: {e}")
                    continue

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
        config_std = np.nanstd(padded, axis=0)
        config_mean_smoothed = smooth_curve(config_mean, window=smooth_window)
        config_std_smoothed = smooth_curve(config_std, window=smooth_window)

        epochs = np.arange(1, len(config_mean_smoothed) + 1)

        # Plot mean with standard deviation as shaded area
        line = plt.plot(epochs, config_mean_smoothed, alpha=0.7, linewidth=1)
        color = line[0].get_color()
        lower = np.clip(config_mean_smoothed - config_std_smoothed, 0.0, 1.0)
        upper = np.clip(config_mean_smoothed + config_std_smoothed, 0.0, 1.0)
        plt.fill_between(epochs, lower, upper, alpha=0.1, color=color)

        final_vals.append(last_valid(config_mean_smoothed))

    plt.title(f"{metric.replace('val_', 'Validation ').title()} Over Epochs\n{dataset_name} ({arch_name}) - All Configs", fontsize=21, fontweight='bold')
    plt.xlabel("Epoch", fontsize=19, fontweight='bold')
    plt.ylabel(metric.replace("_", " ").title(), fontsize=19, fontweight='bold')
    plt.grid(True)
    
    # Increase tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Set x-axis limits to dataset maximum across all models
    plt.xlim(1, dataset_max_epochs)
    
    # Set y-axis limits with dataset-specific scaling (same as representative curves but no baseline)
    if "accuracy" in metric.lower():
        # For accuracy metrics, use dataset-specific scaling with padding
        dataset_max = get_dataset_max_accuracy(dataset_name, metric)
        # Add padding above the max value
        if dataset_max < 0.95:
            # For lower accuracies, add 10% padding
            upper_limit = dataset_max + 0.1 * (dataset_max - 0.0)
        else:
            # For high accuracies (>0.95), add more fixed padding to avoid touching the border
            upper_limit = min(1.08, dataset_max + 0.05)  # Allow more overshoot above 1.0 for better padding
        plt.ylim(0, upper_limit)
    else:
        # For non-accuracy metrics, use standard 0-1 range
        plt.ylim(0, 1)
    
    # legend for standard deviation
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.7, linewidth=1, label='Mean'),
        Patch(facecolor='gray', alpha=0.4, label='±1 Standard Deviation')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=16)

    text = f"{len(grouped)} configs × {len(runs)} seeds × {len(seed_curves) // len(runs)} folds"
    plt.text(0.01, 0.01, text, transform=plt.gca().transAxes, fontsize=16, color='gray', ha='left', va='bottom')

    filename = f"{dataset_name}_{metric}_{arch_name}.pdf"
    save_path = os.path.join("learning_curves", "plots", arch_name, dataset_name, filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
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
        config_std = np.nanstd(padded, axis=0)
        config_mean_smoothed = smooth_curve(config_mean, window=smooth_window)
        config_std_smoothed = smooth_curve(config_std, window=smooth_window)

        def best_loss_score(curve, warmup=10):
            curve = np.array(curve)
            if np.all(np.isnan(curve)):
                return np.inf
            return np.nanmin(curve[warmup:])

        if metric.endswith("loss"):
            final_val = best_loss_score(config_mean_smoothed)
            reach_epoch = np.nanargmin(config_mean_smoothed)
            stability = get_stability(config_mean_smoothed, reach_epoch)
            score = (final_val, -reach_epoch, stability)
        else:
            final_val = last_valid(config_mean_smoothed)
            reach_epoch = get_reach_epoch(config_mean_smoothed, final_val)
            stability = get_stability(config_mean_smoothed, reach_epoch)
            score = (final_val, -reach_epoch, stability)

        config_curves.append((config_id, score, config_mean_smoothed, config_std_smoothed))

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
        "Worst": "#E41A1C",     # bright red
        "1st Quartile": "#FF7F00",  # orange
        "Median": "#4DAF4A",    # green
        "3rd Quartile": "#377EB8",  # blue
        "Best": "#984EA3"       # purple
    }
    
    markers = {
        "Worst": "v",      # triangle down
        "1st Quartile": "s",  # square
        "Median": "o",     # circle
        "3rd Quartile": "^",  # triangle up
        "Best": "*"        # star
    }

    plt.figure(figsize=(12, 6))
    for label, idx in indices.items():
        config_id, _, curve, std_curve = config_curves[idx]
        epochs = np.arange(1, len(curve) + 1)
        line = plt.plot(epochs, curve, label=f"{label} (config_{config_id})", 
                       color=colors[label], linewidth=2, marker=markers[label], 
                       markersize=6, markevery=max(1, len(epochs)//15))
        lower = np.clip(curve - std_curve, 0.0, 1.0)
        upper = np.clip(curve + std_curve, 0.0, 1.0)
        plt.fill_between(epochs, lower, upper, alpha=0.1, color=colors[label])

    # Set x-axis limits to dataset maximum across all models
    plt.xlim(1, dataset_max_epochs)

    # Add baseline for accuracy metrics
    if "accuracy" in metric.lower():
        baseline = get_majority_class_baseline(dataset_name)
        if baseline is not None:
            plt.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, 
                       alpha=0.7, label=f'Majority Class Baseline ({baseline:.2f})')
        
        # Set y-axis limits based on dataset-specific maximum with padding
        dataset_max = get_dataset_max_accuracy(dataset_name, metric)
        # Add padding above the max value
        if dataset_max < 0.95:
            # For lower accuracies, add 10% padding
            upper_limit = dataset_max + 0.1 * (dataset_max - 0.0)
        else:
            # For high accuracies (>0.95), add more fixed padding to avoid touching the border
            upper_limit = min(1.08, dataset_max + 0.05)  # Allow more overshoot above 1.0 for better padding
        plt.ylim(0, upper_limit)

    plt.title(f"Representative Configs ({dataset_name} - {arch_name})\n{metric.replace('val_', 'Validation ').title()}", fontsize=21, fontweight='bold')
    plt.xlabel("Epoch", fontsize=19, fontweight='bold')
    plt.ylabel(metric.replace("_", " ").title(), fontsize=19, fontweight='bold')
    plt.grid(True)
    
    # Increase tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Get existing legend handles and labels, then add std dev info
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(Patch(facecolor='gray', alpha=0.4, label='±1 Standard Deviation'))
    labels.append('±1 Standard Deviation')
    plt.legend(handles=handles, labels=labels, loc='lower right', fontsize=16)

    rep_dir = os.path.join("learning_curves", "representative_curves", arch_name, dataset_name)
    os.makedirs(rep_dir, exist_ok=True)
    rep_filename = f"{dataset_name}_{metric}_{arch_name}_representative.pdf"
    rep_path = os.path.join(rep_dir, rep_filename)
    plt.tight_layout()
    plt.savefig(rep_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Representative curves saved to {rep_path}")

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