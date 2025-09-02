import os
import pandas as pd

result_dirs = {
    "XGBoost": "xgboost_results",
    "RandomForest": "random_forest_results",
    "LinearRegression": "linear_regression_results"
}

all_results = []

for regressor, base_dir in result_dirs.items():
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Skipping.")
        continue

    for model_name in os.listdir(base_dir):
        result_path = os.path.join(base_dir, model_name, f"{model_name}_results.csv")
        if not os.path.exists(result_path):
            continue

        try:
            df = pd.read_csv(result_path)
            df["Regressor"] = regressor
            all_results.append(df)
        except Exception as e:
            print(f"Failed to read {result_path}: {e}")

if not all_results:
    print("No results found. Exiting.")
    exit()

merged = pd.concat(all_results, ignore_index=True)

expected_cols = ["Model", "Dataset", "Regressor", "R2", "R2_std", "RMSE", "RMSE_std", "MAE", "MAE_std"]
merged = merged[[col for col in expected_cols if col in merged.columns]]

merged["R2 ± std"] = merged.apply(lambda row: f"{row['R2']:.2f} ± {row['R2_std']:.2f}", axis=1)
merged["RMSE ± std"] = merged.apply(lambda row: f"{row['RMSE']:.3f} ± {row['RMSE_std']:.3f}", axis=1)
merged["MAE ± std"] = merged.apply(lambda row: f"{row['MAE']:.3f} ± {row['MAE_std']:.3f}", axis=1)

# summary DataFrame with only relevant columns
summary = merged[["Model", "Dataset", "Regressor", "R2 ± std", "RMSE ± std", "MAE ± std"]]

# sorts the summary DataFrame
summary = summary.sort_values(by=["Model", "Dataset", "Regressor"]).reset_index(drop=True)

# saves the summary DataFrame to a CSV file
output_path = "surrogate_model_comparison.csv"
summary.to_csv(output_path, index=False)
print(f"\n Combined surrogate model results saved to: {output_path}")

print("\n Preview:")
print(summary.head(10).to_string(index=False))