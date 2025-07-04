import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay

RESULTS_FILE = "random_forest_results.csv"

# write header if file does not exist
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "Regressor", "R2", "RMSE", "MAE"])

def process_csv(csv_path, model_name, dataset_name):
    print(f"\nProcessing: {model_name}/{dataset_name}")

    try:
        df = pd.read_csv(csv_path)
        if len(df) < 5:
            print(f"Not enough total rows in {csv_path}. Skipping.")
            return
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return

    if "val_acc_20" not in df.columns:
        print(f"'val_acc_20' missing in {csv_path}. Skipping.")
        return

    hp_cols = [col for col in df.columns if col not in ["dataset"] and col.startswith(("dropout", "learning_rate", "num_", "activation", "kernel", "pooling", "bidirectional", "weight_decay", "ff_dim"))]

    if not hp_cols:
        print(f"No hyperparameter columns in {csv_path}. Skipping.")
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

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train random forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # log results
    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, dataset_name, "RandomForest", round(r2, 4), round(rmse, 4), round(mae, 4)])

    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    plot_base = os.path.join("random_forest_plots", model_name)
    os.makedirs(plot_base, exist_ok=True)

    # --- Plot 1: Feature Importance ---
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
    fi_path = os.path.join(fi_dir, f"{model_name}_{dataset_name}_feat_imp.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance to: {fi_path}")

    # --- Plot 2: Partial Dependence Plot (top 2 features) ---
    top2 = indices[:2].tolist()
    features = [feat_names[i] for i in top2]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    PartialDependenceDisplay.from_estimator(model, X_train, features, ax=ax)
    plt.suptitle(f"{model_name}/{dataset_name} – Partial Dependence (top 2 features)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pdp_dir = os.path.join(plot_base, "partial_dependence")
    os.makedirs(pdp_dir, exist_ok=True)
    pdp_path = os.path.join(pdp_dir, f"{model_name}_{dataset_name}_pdp.png")
    plt.savefig(pdp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PDP to: {pdp_path}")

    # --- Plot 3: Actual vs Predicted Scatter ---
    plt.figure(figsize=(8, 6))
    y_test_pct = y_test * 100
    y_pred_pct = y_pred * 100

    plt.scatter(y_test_pct, y_pred_pct, color='#1f77b4', alpha=0.7, s=80, edgecolor='k', linewidth=0.5, label="Predicted vs Actual")

    min_val = min(y_test_pct.min(), y_pred_pct.min()) - 2
    max_val = max(y_test_pct.max(), y_pred_pct.max()) + 2

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label="Perfect Prediction")
    plt.xlabel("Actual Validation Accuracy (%)", fontsize=12)
    plt.ylabel("Predicted Validation Accuracy (%)", fontsize=12)
    plt.title(f"{model_name} - {dataset_name}\nRandom Forest Prediction", fontsize=14, pad=20)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right') 

    textstr = f'R² = {r2:.2f}\nRMSE = {rmse:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)

    pred_path = os.path.join(plot_base, f"{model_name}_{dataset_name}_rf.png")
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
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

            for fname in os.listdir(dataset_path):
                if fname.endswith(".csv"):
                    csv_path = os.path.join(dataset_path, fname)
                    process_csv(csv_path, model_name, dataset_name)
