import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

# this function evaluates how well a surrogate model performs
def evaluate_model(model, X, y, seeds=20):
    preds, actuals = [], []
    r2s, rmses, maes = [], [], []

    for seed in range(seeds):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        preds.extend(y_pred)
        actuals.extend(y_val)

        r2s.append(r2_score(y_val, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        maes.append(mean_absolute_error(y_val, y_pred))

    return {
        "r2": (np.mean(r2s), np.std(r2s)),
        "rmse": (np.mean(rmses), np.std(rmses)),
        "mae": (np.mean(maes), np.std(maes)),
        "preds": np.array(preds),
        "actuals": np.array(actuals)
    }

# this function plots combined results for all regressors
def plot_combined(actuals_list, preds_list, scores_list, model_labels, model_name, dataset_name):
    colors = ['teal', 'steelblue', 'firebrick']
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, ax in enumerate(axs):
        ax.scatter(actuals_list[i], preds_list[i], alpha=0.6, edgecolors='k', s=70, color=colors[i])
        min_val = min(actuals_list[i].min(), preds_list[i].min())
        max_val = max(actuals_list[i].max(), preds_list[i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

        ax.set_xlabel("Actual Validation Accuracy", fontsize=19, fontweight='bold')
        ax.set_ylabel("Predicted Accuracy", fontsize=19, fontweight='bold')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
        
        ax.tick_params(axis='both', which='major', labelsize=18)

        r2, rmse, mae = scores_list[i]['r2'], scores_list[i]['rmse'], scores_list[i]['mae']
        textstr = f"$R^2$: {r2[0]:.2f} ± {r2[1]:.2f}\nRMSE: {rmse[0]:.3f} ± {rmse[1]:.3f}\nMAE: {mae[0]:.3f} ± {mae[1]:.3f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=16, va='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))
        ax.set_title(f"{model_name} - {dataset_name}\n{model_labels[i]}", fontsize=21, fontweight='bold')

    plt.tight_layout()
    outdir = os.path.join("regressor_plots", model_name)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{model_name}_{dataset_name}_combined.pdf")
    plt.savefig(outpath, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved combined plot to: {outpath}")

# this function processes a CSV file for a specific model and dataset
def process_csv(path, model_name, dataset_name):
    df = pd.read_csv(path)
    if "val_acc_20" not in df.columns or len(df) < 5:
        return

    hp_cols = [col for col in df.columns if col.startswith(("dropout", "learning_rate", "num_", "activation", "kernel", "pooling", "bidirectional", "weight_decay", "ff_dim"))]
    if not hp_cols:
        return

    X = df[hp_cols]
    y = df["val_acc_20"].dropna()
    X = X.loc[y.index]

    X = pd.get_dummies(X)
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    print(f"Processing: {model_name}/{dataset_name} - Samples: {len(X)}")

    results = []
    regressors = [
        ("Linear Regression Surrogate", LinearRegression()),
        ("Random Forest Surrogate", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("XGBoost Surrogate", XGBRegressor(n_estimators=100, verbosity=0, random_state=42))
    ]

    for name, reg in regressors:
        res = evaluate_model(reg, X, y)
        results.append(res)

    preds = [res['preds'] for res in results]
    actuals = [res['actuals'] for res in results]
    scores = results

    plot_combined(actuals, preds, scores, [r[0] for r in regressors], model_name, dataset_name)

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
