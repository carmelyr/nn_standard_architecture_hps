import os
import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

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

    # --- 20x Holdout Evaluation --- #
    for seed in range(20):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

        model = XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))

        all_preds.extend(y_pred)
        all_actuals.extend(y_val)

    # final mean ± std values
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
    mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)

    print("XGBoost Results (20x 80-20 Holdout):")
    print(f"R²:   {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"MAE:  {mae_mean:.4f} ± {mae_std:.4f}")

    # saves CSV summary
    results_dir = os.path.join("xgboost_results", model_name)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}_results.csv")

    write_header = not os.path.exists(results_file)
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Dataset", "Regressor", "R2", "R2_std", "RMSE", "RMSE_std", "MAE", "MAE_std"])
        writer.writerow([model_name, dataset_name, "XGBoost", round(r2_mean, 4), round(r2_std, 4), round(rmse_mean, 4), round(rmse_std, 4), round(mae_mean, 4), round(mae_std, 4)])

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    plt.figure(figsize=(7, 7))
    plt.scatter(all_actuals, all_preds, color='FireBrick', alpha=0.5, edgecolors='k', s=70)

    min_val = min(all_actuals.min(), all_preds.min())
    max_val = max(all_actuals.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

    textstr = '\n'.join((f'$R^2$: {r2_mean:.2f} ± {r2_std:.2f}', f'RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}', f'MAE: {mae_mean:.2f} ± {mae_std:.2f}'))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

    plt.xlabel("Actual Validation Accuracy")
    plt.ylabel("Predicted Accuracy")
    plt.title(f"{model_name} - {dataset_name}\nXGBoost Surrogate")
    plt.grid(True)
    plt.tight_layout()

    model_plot_dir = os.path.join("xgboost_plots", model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    plot_path = os.path.join(model_plot_dir, f"{model_name}_{dataset_name}_xgb.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")

    # --- Generate learning curves --- #
    train_sizes = [20, 40, 60, 80]
    metrics_per_size = {size: {'r2': [], 'rmse': [], 'mae': []} for size in train_sizes}

    # --- 20x Holdout Learning Curves --- #
    for seed in range(20):
        X_shuffled, y_shuffled = shuffle(X, y, random_state=seed)

        for size in train_sizes:
            if size > len(X_shuffled) - 20:
                continue

            X_train = X_shuffled[:size]
            y_train = y_shuffled[:size]
            X_val = X_shuffled[-20:]
            y_val = y_shuffled[-20:]

            model = XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            metrics_per_size[size]['r2'].append(r2_score(y_val, y_pred))
            metrics_per_size[size]['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            metrics_per_size[size]['mae'].append(mean_absolute_error(y_val, y_pred))

    # --- Plot learning curves ---
    """fig, ax = plt.subplots(3, 1, figsize=(7, 12))
    metrics = ['r2', 'rmse', 'mae']
    titles = ['R² Score', 'RMSE', 'MAE']

    for i, metric in enumerate(metrics):
        means = [np.mean(metrics_per_size[size][metric]) for size in train_sizes]
        stds = [np.std(metrics_per_size[size][metric]) for size in train_sizes]

        ax[i].errorbar(train_sizes, means, yerr=stds, label="XGBoost", marker='o', capsize=4)
        ax[i].set_xlabel("Number of Training Examples")
        ax[i].set_ylabel(titles[i])
        ax[i].set_title(f"{titles[i]} vs Training Size — {dataset_name}")
        ax[i].grid(True)

    plt.tight_layout()
    curve_plot_dir = os.path.join("xgboost_learning_curves", model_name)
    os.makedirs(curve_plot_dir, exist_ok=True)
    curve_path = os.path.join(curve_plot_dir, f"{model_name}_{dataset_name}_learning_curve.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to: {curve_path}")"""

    # --- Save learning curve data to JSON --- #
    learning_curve_data = {
        "dataset": dataset_name,
        "regressor": "XGBoost",
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
    json_path = os.path.join(curve_json_dir, f"XGBoost_{dataset_name}.json")

    with open(json_path, "w") as f:
        json.dump(learning_curve_data, f, indent=2)

    print(f"Saved learning curve data to: {json_path}")

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
