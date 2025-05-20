import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

# this method is used to load the results from the JSON files
# reads each JSON file (one per run/config)
# extracts the dataset name, classifier name, and validation accuracy
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

# helper function for sorting categorical values (like string-binned hyperparameters) in numeric order
def safe_float_str(x):
    try:
        if isinstance(x, str) and '-' in x:
            return float(x.split('-')[0])
        return float(x)
    except:
        return float('inf')

# this function generates a 3D heatmap of the validation accuracy
# X-axis: classifiers (model types)
# Y-axis: datasets
# Z-axis: hyperparameter values (binned)
def generate_3d_heatmap(df, metric="max_val_accuracy", hyperparam="learning_rate", classifiers=None):
    df = df.copy()
    is_numeric = pd.api.types.is_numeric_dtype(df[hyperparam])

    # bins the selected hyperparameter into 10 intervals.
    # groups the data by dataset + classifier + hyperparameter bin
    if is_numeric:
        num_bins = 10
        int_bins = ["num_filters", "hidden_units", "num_layers", "ff_dim", "num_heads", "kernel_size", "pooling_size"]

        # bins only the non-null values
        raw_vals = df[hyperparam].dropna()
        bin_categories = pd.cut(raw_vals, bins=num_bins, include_lowest=True).cat.categories

        if hyperparam in int_bins:
            bin_labels = [f"{int(b.left)}-{int(b.right)}" for b in bin_categories]
            label_func = lambda x: f"{int(x.left)}-{int(x.right)}"
        else:
            bin_labels = [f"{b.left:.3f}-{b.right:.3f}" for b in bin_categories]
            label_func = lambda x: f"{x.left:.3f}-{x.right:.3f}"

        # fills in missing combinations with NaN (to preserve full grid structure)
        df["_bin"] = pd.cut(df[hyperparam], bins=bin_categories)
        df[hyperparam] = df["_bin"].apply(lambda x: label_func(x) if pd.notnull(x) else None)
        del df["_bin"]

        hp_values = bin_labels

    else:
        df[hyperparam] = df[hyperparam].astype(str)
        hp_values = sorted(df[hyperparam].dropna().unique(), key=safe_float_str)

    # prepares categories
    datasets = sorted(df["dataset"].unique())

    if classifiers is None:
        classifiers = sorted(df["classifier"].unique())

    df["dataset_idx"] = df["dataset"].apply(lambda x: datasets.index(x))
    df["classifier_idx"] = df["classifier"].apply(lambda x: classifiers.index(x))


    fig = plt.figure(figsize=(max(12, len(classifiers)*0.8), max(8, len(datasets)*0.6)))
    ax = fig.add_subplot(111, projection='3d')

    # forces invisible 0-height bars for layout preservation
    for clf_idx in range(len(classifiers)):
        ax.bar3d(
            x=clf_idx, y=0, z=0,
            dx=0.01, dy=0.01, dz=0.0,  # tiny invisible bar
            color=(0, 0, 0, 0),        # fully transparent
            alpha=0.0
        )


    norm = Normalize(vmin=df[metric].min(), vmax=df[metric].max())
    cmap = cm.get_cmap("RdPu")
    layer_spacing = 2.0

    for i, hp_val in enumerate(hp_values):
        for xi in range(len(classifiers)):
            ax.bar3d(
                x=xi, y=0, z=i * layer_spacing,
                dx=0.7, dy=0.8, dz=0.00001,
                color=(0, 0, 0, 0),
                edgecolor=None,
                linewidth=0,
                alpha=0.0
            )

        # extracts current layer
        layer = df[df[hyperparam] == hp_val].copy()

        # injects rows for classifiers that are completely missing this hyperparameter bin
        for clf in classifiers:
            if clf not in layer["classifier"].values:
                # adds a dummy row for each dataset for this classifier
                for dataset in datasets:
                    layer = pd.concat([
                        layer,
                        pd.DataFrame([{
                            "dataset": dataset,
                            "classifier": clf,
                            "dataset_idx": datasets.index(dataset),
                            "classifier_idx": classifiers.index(clf),
                            metric: np.nan
                        }])
                    ])


        index_range = pd.Index(range(len(datasets)), name="dataset_idx")
        columns_range = pd.Index(range(len(classifiers)), name="classifier_idx")

        pivot = (layer.pivot_table(index="dataset_idx", columns="classifier_idx", values=metric, aggfunc='mean').reindex(index=index_range, columns=columns_range, fill_value=np.nan))

        X, Y = np.meshgrid(
            np.arange(len(classifiers)),
            np.arange(len(datasets))
        )
        Z = np.full_like(X, i * layer_spacing)

        dz = pivot.values
        C = norm(np.nan_to_num(dz, nan=0.0))  # convert NaNs for color mapping
        colors = cmap(C)

        for xi in range(len(classifiers)):
            for yi in range(len(datasets)):
                val = dz[yi, xi]
                if pd.notna(val):
                    ax.bar3d(
                        x=xi, y=yi, z=i * layer_spacing,
                        dx=0.7, dy=0.8, dz=val,
                        color=colors[yi, xi],
                        edgecolor=None,
                        linewidth=0,
                        alpha=0.8
                    )
                else:
                    # draws a zero-height transparent bar to reserve the spot
                    ax.bar3d(
                        x=xi, y=yi, z=i * layer_spacing,
                        dx=0.01, dy=0.01, dz=0.0,
                        color=(0, 0, 0, 0),
                        alpha=0.0
                    )

    ax.set_xlabel("Classifier", labelpad=15, fontsize=12)
    ax.set_ylabel("Dataset", labelpad=40, fontsize=12)
    ax.set_zlabel(hyperparam, labelpad=30, fontsize=12)

    ax.set_xticks(np.arange(len(classifiers)))
    ax.set_xticklabels(classifiers, rotation=0, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(datasets)) + 0.5)
    ax.set_yticklabels(datasets, fontsize=6, rotation=35)
    ax.tick_params(axis='y', labelsize=9, pad=6)
    ax.tick_params(axis='z', labelsize=9, pad=10)

    z_ticks = np.arange(len(hp_values)) * layer_spacing
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(hp_values, fontsize=10)

    for label in ax.zaxis.get_ticklabels():
        label.set_rotation(30)


    ax.view_init(elev=25, azim=45)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(df[metric])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=30, pad=0.1, format='%.2f')
    cbar.set_label(metric, rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.title(f"3D Heatmap of {metric} by Dataset, Classifier and {hyperparam} (10 bins)", pad=25, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()

def select_hyperparameter(available_hyperparams):
    print("\nAvailable hyperparameters:")
    for i, hp in enumerate(available_hyperparams, 1):
        print(f"{i}. {hp}")
    while True:
        try:
            choice = int(input(f"\nSelect hyperparameter by number (1-{len(available_hyperparams)}): "))
            if 1 <= choice <= len(available_hyperparams):
                return available_hyperparams[choice - 1]
            print("Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    df = load_results()
    all_classifiers = sorted(df["classifier"].unique())
    hyperparams = [col for col in df.columns
                   if col not in ['dataset', 'classifier', 'max_val_accuracy', 'avg_val_accuracy']]
    
    if not hyperparams:
        print("No hyperparameters found in the data.")
    else:
        selected_hparam = select_hyperparameter(hyperparams)
        generate_3d_heatmap(df, metric="max_val_accuracy", hyperparam=selected_hparam, classifiers=all_classifiers)
