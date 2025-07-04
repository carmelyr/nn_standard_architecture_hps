import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def load_and_combine_results():
    files = {
        'LinearRegression': 'linear_regression_results.csv',
        'RandomForest': 'random_forest_results.csv',
        'XGBoost': 'xgboost_results.csv'
    }

    dfs = []
    for regressor, filename in files.items():
        df = pd.read_csv(filename)
        df['Regressor'] = regressor
        dfs.append(df)

    combined_df = pd.concat(dfs)
    combined_df = combined_df.drop_duplicates(subset=["Model", "Dataset", "Regressor"])
    return combined_df

# this function creates pairwise heatmaps comparing the average performance of different regressors over all datasets
def generate_pairwise_summary_heatmaps(combined_df):
    metrics = ['R2', 'RMSE', 'MAE']
    regressors = ['LinearRegression', 'RandomForest', 'XGBoost']
    model_archs = combined_df['Model'].unique()

    for model_arch in model_archs:
        df_model = combined_df[combined_df["Model"] == model_arch]

        if df_model["Regressor"].nunique() < 3:
            continue

        out_dir = f"surrogate_heatmaps_summary/{model_arch}"
        os.makedirs(out_dir, exist_ok=True)

        for metric in metrics:
            pivot = df_model.pivot(index="Dataset", columns="Regressor", values=metric)

            if pivot.isnull().any().any():
                continue

            # builds 3x3 matrix of average metric differences between regressors
            heatmap_data = pd.DataFrame(index=regressors, columns=regressors, dtype=float)

            for reg_a in regressors:
                for reg_b in regressors:
                    if reg_a == reg_b:
                        heatmap_data.loc[reg_a, reg_b] = 0
                    else:
                        # Regressor A better → positive (red), Regressor B better → negative (blue)
                        if metric in ["RMSE", "MAE"]:
                            diff = (pivot[reg_b] - pivot[reg_a]).mean()     # lower RMSE/MAE is better
                        else:
                            diff = (pivot[reg_a] - pivot[reg_b]).mean()     # higher R² is better
                        heatmap_data.loc[reg_a, reg_b] = diff

            # plots the heatmap
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(heatmap_data.astype(float), cmap="coolwarm", center=0, annot=True, fmt=".3f", annot_kws={"size": 10}, linewidths=0.5, square=True, cbar=True)
            ax.collections[0].colorbar.set_label(f"{metric} Difference (A - B)", fontsize=10)
            plt.title(f"{model_arch} - {metric} Avg Difference (Regressor vs. Regressor)")
            plt.suptitle("Red = A better, Blue = B better", fontsize=8, y=0.98)
            plt.xlabel("Regressor B", fontweight='bold')
            plt.ylabel("Regressor A", fontweight='bold')
            plt.tight_layout()

            filename = f"{metric}_summary_heatmap.png"
            plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Saved summary heatmap to: {os.path.join(out_dir, filename)}")


# this function generates pairwise heatmaps for each dataset and model architecture
def generate_dataset_pairwise_heatmaps(combined_df):
    metrics = ['R2', 'RMSE', 'MAE']
    regressors = ['LinearRegression', 'RandomForest', 'XGBoost']
    model_archs = combined_df['Model'].unique()
    datasets = combined_df['Dataset'].unique()

    for model_arch in model_archs:
        df_model = combined_df[combined_df["Model"] == model_arch]

        if df_model["Regressor"].nunique() < 3:
            continue

        for metric in metrics:
            for dataset in datasets:
                df_dataset = df_model[df_model["Dataset"] == dataset]
                if df_dataset["Regressor"].nunique() < 3:
                    continue

                diff_matrix = pd.DataFrame(index=regressors, columns=regressors)

                for reg_a in regressors:
                    for reg_b in regressors:
                        try:
                            val_a = df_dataset[df_dataset['Regressor'] == reg_a][metric].values[0]
                            val_b = df_dataset[df_dataset['Regressor'] == reg_b][metric].values[0]

                            if metric in ["RMSE", "MAE"]:
                                diff = val_b - val_a        # lower RMSE/MAE is better
                            else:
                                diff = val_a - val_b        # higher R² is better

                            diff_matrix.loc[reg_a, reg_b] = diff
                        except IndexError:
                            diff_matrix.loc[reg_a, reg_b] = None

                if diff_matrix.isnull().values.any():
                    continue

                diff_matrix = diff_matrix.astype(float)

                out_dir = f"surrogate_heatmaps_pairwise/{model_arch}/{dataset}"
                os.makedirs(out_dir, exist_ok=True)

                plt.figure(figsize=(6, 5))
                ax2 = sns.heatmap(diff_matrix, cmap="coolwarm", center=0, annot=True, fmt=".3f", cbar=True, square=True)
                ax2.collections[0].colorbar.set_label(f"{metric} Difference (A - B)", fontsize=10)
                plt.title(f"{model_arch} - {dataset} - {metric} (Regressor A - B)")
                plt.suptitle("Red = A better, Blue = B better", fontsize=8, y=0.98)
                plt.xlabel("Regressor B", fontweight='bold')
                plt.ylabel("Regressor A", fontweight='bold')
                plt.tight_layout()
                filename = f"{metric}_pairwise_heatmap.png"
                plt.savefig(os.path.join(out_dir, filename), dpi=300)
                plt.close()
                print(f"Saved pairwise heatmap to: {os.path.join(out_dir, filename)}")


if __name__ == "__main__":
    print("Loading and combining results...")
    combined_df = load_and_combine_results()
    print("Generating heatmaps...")
    generate_pairwise_summary_heatmaps(combined_df)
    print("Generating per-dataset pairwise heatmaps...")
    generate_dataset_pairwise_heatmaps(combined_df)
    print("All heatmaps generated successfully!")
