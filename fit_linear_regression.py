import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

RESULTS_FILE = "linear_regression_results.csv"

# writes a header if file does not exist yet
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "Regressor", "R2", "RMSE", "MAE"])

# this function processes each CSV file from the surrogate_datasets directory
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

    print(f"Total rows in CSV: {len(df)}")

    # one-hot encodes and imputes missing values
    X = pd.get_dummies(X)
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    print(f"Rows after cleaning: {len(X)}")

    if len(X) < 2:
        print("Not enough samples after cleaning. Skipping.")
        return

    # train-test split with 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train model using linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculates R² and RMSE
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # logs results
    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model_name, dataset_name, "LinearRegression", round(r2, 4), round(rmse, 4), round(mae, 4)])

    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # regression fit line
    fit_model = LinearRegression()
    fit_model.fit(y_test.values.reshape(-1, 1), y_pred)
    x_line = np.linspace(y_test.min(), y_test.max(), 100).reshape(-1, 1)
    y_line = fit_model.predict(x_line)

    # plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, color='DeepPink', alpha=0.6, label="Predicted Points")
    plt.plot(x_line, y_line, color='darkslateblue', label="Regression Fit")
    plt.xlabel("True Accuracy at Epoch 20")
    plt.ylabel("Predicted Accuracy")
    plt.title(f"{model_name}/{dataset_name}")
    plt.legend()
    plt.grid(True)

    # saves the plots
    model_plot_dir = os.path.join("linear_regression_plots", model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    plot_path = os.path.join(model_plot_dir, f"{model_name}_{dataset_name}_linear.png")
    plt.tight_layout()
    plt.savefig(plot_path)
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

            for fname in os.listdir(dataset_path):
                if fname.endswith(".csv"):
                    csv_path = os.path.join(dataset_path, fname)
                    process_csv(csv_path, model_name, dataset_name)
