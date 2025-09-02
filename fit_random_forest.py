import os
import pandas as pd
import numpy as np
import csv
from sklearn.utils import shuffle
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

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

    if len(df) < 5:
        print(f"Not enough total rows after concatenation. Skipping.")
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

    if len(X) < 2:
        print("Not enough samples after cleaning. Skipping.")
        return

    # --- 20x Holdout Evaluation --- #
    r2_scores, rmse_scores, mae_scores = [], [], []
    all_preds, all_actuals = [], []

    train_sizes = [20, 40, 60, 80]
    metrics_per_size = {size: {'r2': [], 'rmse': [], 'mae': []} for size in train_sizes}

    for seed in range(20):
        X_shuffled, y_shuffled = shuffle(X, y, random_state=seed)

        for size in train_sizes:
            if size > len(X_shuffled) - 20:
                continue

            X_train = X_shuffled[:size]
            y_train = y_shuffled[:size]
            X_val = X_shuffled[-20:]
            y_val = y_shuffled[-20:]

            model = RandomForestRegressor(n_estimators=100, random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            metrics_per_size[size]['r2'].append(r2_score(y_val, y_pred))
            metrics_per_size[size]['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            metrics_per_size[size]['mae'].append(mean_absolute_error(y_val, y_pred))

    # --- Saves learning curve data to JSON --- #
    learning_curve_data = {
        "dataset": dataset_name,
        "regressor": "RandomForest",
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
    json_path = os.path.join(curve_json_dir, f"RandomForest_{dataset_name}.json")

    with open(json_path, "w") as f:
        json.dump(learning_curve_data, f, indent=2)

    print(f"Saved learning curve data to: {json_path}")

    # --- Final evaluation on the full dataset --- #
    for seed in range(20):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))

        all_preds.extend(y_pred)
        all_actuals.extend(y_val)

    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)
    rmse_mean, rmse_std = np.mean(rmse_scores), np.std(rmse_scores)
    mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)

    print(f"R²:   {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"MAE:  {mae_mean:.4f} ± {mae_std:.4f}")

    # saves results to CSV
    results_dir = os.path.join("random_forest_results", model_name)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name}_results.csv")

    write_header = not os.path.exists(results_file)
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Dataset", "Regressor", "R2", "R2_std", "RMSE", "RMSE_std", "MAE", "MAE_std"])
        writer.writerow([model_name, dataset_name, "RandomForest", round(r2_mean, 4), round(r2_std, 4), round(rmse_mean, 4), round(rmse_std, 4), round(mae_mean, 4), round(mae_std, 4)])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    plot_base = os.path.join("random_forest_plots", model_name)
    os.makedirs(plot_base, exist_ok=True)

    # --- Plot 1: Feature Importance --- #
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name}/{dataset_name} – Feature Importances", fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color='#2ca02c', edgecolor='k')
    plt.yticks(range(len(indices)), feat_names[indices])
    plt.xlabel("Importance (mean decrease in impurity)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    fi_dir = os.path.join(plot_base, "feature_importance")
    os.makedirs(fi_dir, exist_ok=True)
    fi_path = os.path.join(fi_dir, f"{model_name}_{dataset_name}_feat_imp.pdf")
    plt.savefig(fi_path, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved feature importance to: {fi_path}")

    # --- Plot 2: Actual vs Predicted Scatter --- #
    plt.figure(figsize=(8, 6))
    plt.scatter(all_actuals, all_preds, color='#1f77b4', alpha=0.7, s=80, edgecolor='k', linewidth=0.5)

    min_val = min(min(all_actuals), min(all_preds)) - 0.01
    max_val = max(max(all_actuals), max(all_preds)) + 0.01
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    plt.xlabel("Actual Validation Accuracy", fontsize=12)
    plt.ylabel("Predicted Validation Accuracy", fontsize=12)
    plt.title(f"{model_name} - {dataset_name}\nRandom Forest Surrogate", fontsize=14, pad=20)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.6)

    textstr = f'$R^2$: {r2_mean:.2f} ± {r2_std:.2f}\nRMSE: {rmse_mean:.3f} ± {rmse_std:.3f}\nMAE: {mae_mean:.3f} ± {mae_std:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)

    pred_path = os.path.join(plot_base, f"{model_name}_{dataset_name}_rf.pdf")
    plt.savefig(pred_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved scatter plot to: {pred_path}")

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
