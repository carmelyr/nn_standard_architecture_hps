import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

RESULTS_FILE = "xgboost_results.csv"

# write header if file does not exist
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "Regressor", "R2", "RMSE", "MAE"])

def process_dataset(model_name, dataset_name, dataset_path):
    print(f"\nProcessing: {model_name}/{dataset_name}")

    all_dfs = []
    for fname in os.listdir(dataset_path):
        if fname.endswith(".csv"):
            csv_path = os.path.join(dataset_path, fname)
            try:
                df = pd.read_csv(csv_path)
                if "val_acc_20" not in df.columns:
                    continue
                if len(df) < 5:
                    continue
                all_dfs.append(df)
            except Exception as e:
                print(f"Failed to read {csv_path}: {e}")

    if not all_dfs:
        print("No valid CSVs found. Skipping.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    hp_cols = [col for col in df.columns if col not in ["dataset"] and col.startswith(("dropout", "learning_rate", "num_", "activation", "kernel", "pooling", "bidirectional", "weight_decay", "ff_dim"))]
    if not hp_cols:
        print(f"No hyperparameter columns in {dataset_name}. Skipping.")
        return

    X = df[hp_cols]
    y = df["val_acc_20"].dropna()
    X = X.loc[y.index]

    print(f"Total rows in combined CSVs: {len(df)}")

    X = pd.get_dummies(X)
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    print(f"Rows after cleaning: {len(X)}")
    if len(X) < 2:
        print("Not enough samples after cleaning. Skipping.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, dataset_name, "XGBoost", round(r2, 4), round(rmse, 4), round(mae, 4)])

    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, color='FireBrick', alpha=0.7, edgecolors='k', s=70, label="Predicted vs Actual")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Prediction")

    textstr = '\n'.join((f'$R^2$: {r2:.2f}', f'RMSE: {rmse:.2f}', f'MAE: {mae:.2f}'))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

    plt.xlabel("Actual Validation Accuracy at Epoch 20")
    plt.ylabel("Predicted Validation Accuracy")
    plt.title(f"{model_name} - {dataset_name}\nXGBoost Prediction", fontsize=13)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    model_plot_dir = os.path.join("xgboost_plots", model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    plot_path = os.path.join(model_plot_dir, f"{model_name}_{dataset_name}_xgb.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")

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
