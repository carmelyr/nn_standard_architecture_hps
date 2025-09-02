import os
import pandas as pd
import numpy as np
import csv
import json
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# this function processes all CSV files for a dataset by concatenating them
def process_dataset(model_name, dataset_name, dataset_path):
    print(f"\nProcessing: {model_name}/{dataset_name}")
    all_dfs = []
    for fname in os.listdir(dataset_path):
        if fname.endswith(".csv"):
            csv_path = os.path.join(dataset_path, fname)
            try:
                df = pd.read_csv(csv_path)
                if "val_acc_20" not in df.columns or len(df) < 5:
                    continue
                all_dfs.append(df)
            except Exception as e:
                print(f"Failed to read {csv_path}: {e}")

    if not all_dfs:
        print("No valid CSVs found. Skipping.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    try:
        if len(df) < 5:
            print(f"Not enough total rows after concatenation. Skipping.")
            return
    except Exception as e:
        print(f"Failed to process concatenated data: {e}")
        return

    if "val_acc_20" not in df.columns:
        print(f"'val_acc_20' missing in concatenated data. Skipping.")
        return

    hp_cols = [col for col in df.columns if col.startswith(("dropout", "learning_rate", "num_", "activation", "kernel", "pooling", "bidirectional", "weight_decay", "ff_dim"))]
    if not hp_cols:
        print(f"No hyperparameter columns in {dataset_name}. Skipping.")
        return

    X = df[hp_cols]
    y = df["val_acc_20"].dropna()
    X = X.loc[y.index]

    X = pd.get_dummies(X)
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    print(f"Rows after cleaning: {len(X)}")
    if len(X) < 2:
        print("Not enough samples after cleaning. Skipping.")
        return

    r2_scores, rmse_scores, mae_scores = [], [], []
    all_preds, all_actuals = [], []

    # --- Learning curve evaluation --- #
    train_sizes = [20, 40, 60, 80]      # training sizes for learning curves
    metrics_per_size = {size: {'r2': [], 'rmse': [], 'mae': []} for size in train_sizes}

    # 20x holdout splits for learning curves
    for seed in range(20):
        # shuffles the dataset for each seed
        X_shuffled, y_shuffled = shuffle(X, y, random_state=seed)

        for size in train_sizes:
            if size > len(X_shuffled) - 20:
                continue

            X_train = X_shuffled[:size]
            y_train = y_shuffled[:size]
            X_val = X_shuffled[-20:]
            y_val = y_shuffled[-20:]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            metrics_per_size[size]['r2'].append(r2_score(y_val, y_pred))
            metrics_per_size[size]['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            metrics_per_size[size]['mae'].append(mean_absolute_error(y_val, y_pred))


    # --- Final evaluation on the full dataset --- #
    for seed in range(20):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))

        all_preds.extend(y_pred)
        all_actuals.extend(y_val)

    # computes mean +- std
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
    mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)

    print(f"R²: {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")

    # saves to CSV
    results_dir = os.path.join("linear_regression_results", model_name)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}_results.csv")

    write_header = not os.path.exists(results_file)
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Dataset", "Regressor", "R2", "R2_std", "RMSE", "RMSE_std", "MAE", "MAE_std"])
        writer.writerow([model_name, dataset_name, "LinearRegression", round(r2_mean, 4), round(r2_std, 4), round(rmse_mean, 4), round(rmse_std, 4), round(mae_mean, 4), round(mae_std, 4)])

    # scatter plot
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    plt.figure(figsize=(7, 7))
    plt.scatter(all_actuals, all_preds, color='DarkCyan', alpha=0.6, edgecolors='k', s=70)

    min_val = min(all_actuals.min(), all_preds.min())
    max_val = max(all_actuals.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

    textstr = '\n'.join((f'$R^2$: {r2_mean:.2f} ± {r2_std:.2f}', f'RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}', f'MAE: {mae_mean:.2f} ± {mae_std:.2f}'))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

    plt.xlabel("Actual Validation Accuracy")
    plt.ylabel("Predicted Accuracy")
    plt.title(f"{model_name} - {dataset_name}\nLinear Regression Surrogate")
    plt.grid(True)
    plt.tight_layout()

    model_plot_dir = os.path.join("linear_regression_plots", model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    plot_path = os.path.join(model_plot_dir, f"{model_name}_{dataset_name}_linear.pdf")
    plt.savefig(plot_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved plot to: {plot_path}")

    # --- Saves learning curve data to JSON --- #
    learning_curve_data = {
        "dataset": dataset_name,
        "regressor": "LinearRegression",
        "train_sizes": train_sizes,
        "r2_mean": [np.mean(metrics_per_size[size]['r2']) for size in train_sizes],
        "r2_std": [np.std(metrics_per_size[size]['r2']) for size in train_sizes],
        "rmse_mean": [np.mean(metrics_per_size[size]['rmse']) for size in train_sizes],
        "rmse_std": [np.std(metrics_per_size[size]['rmse']) for size in train_sizes],
        "mae_mean": [np.mean(metrics_per_size[size]['mae']) for size in train_sizes],
        "mae_std": [np.std(metrics_per_size[size]['mae']) for size in train_sizes],
    }

    curve_json_dir = os.path.join("learning_curve_data", model_name)
    os.makedirs(curve_json_dir, exist_ok=True)
    json_path = os.path.join(curve_json_dir, f"LinearRegression_{dataset_name}.json")

    with open(json_path, "w") as f:
        json.dump(learning_curve_data, f, indent=2)

    print(f"Saved learning curve data to: {json_path}")

    # --- Feature Importance --- #
    # trains LR on full data for interpretability plots
    model_full = LinearRegression()
    model_full.fit(X, y)

    # Feature Importance (Coefficients) Plot (normalized)
    coefs = model_full.coef_
    coefs_abs = np.abs(coefs)

    # normalizes to sum to 1 so the scale
    total = coefs_abs.sum()
    if total == 0:
        importances = np.zeros_like(coefs_abs)
    else:
        importances = coefs_abs / total

    indices = np.argsort(importances)[::-1]
    feat_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name}/{dataset_name} – Feature Importances", fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color='#2ca02c', edgecolor='k')
    plt.yticks(range(len(indices)), feat_names[indices])
    plt.xlabel("Importance (mean decrease in impurity)")
    plt.gca().invert_yaxis()

    plt.xlim(0, 1)
    plt.tight_layout()

    fi_dir = os.path.join(model_plot_dir, "feature_importance")
    os.makedirs(fi_dir, exist_ok=True)
    fi_path = os.path.join(fi_dir, f"{model_name}_{dataset_name}_feat_imp.pdf")
    plt.savefig(fi_path, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved feature importance to: {fi_path}")

if __name__ == "__main__":
    BASE_DIR = "surrogate_datasets"
    for model_name in os.listdir(BASE_DIR):
        model_path = os.path.join(BASE_DIR, model_name)
        if not os.path.isdir(model_path):
            continue

        for dataset_name in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            process_dataset(model_name, dataset_name, dataset_path)
