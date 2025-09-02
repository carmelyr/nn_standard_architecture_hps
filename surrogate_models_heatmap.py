import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_combine_results():
    base_dirs = {
        'LinearRegression': 'linear_regression_results',
        'RandomForest': 'random_forest_results',
        'XGBoost': 'xgboost_results'
    }

    dfs = []    # list of DataFrames
    for regressor, base_dir in base_dirs.items():
        if not os.path.exists(base_dir):
            print(f"Warning: base directory '{base_dir}' does not exist.")
            continue

        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.endswith("_results.csv"):
                    fpath = os.path.join(root, fname)
                    try:
                        df = pd.read_csv(fpath)
                        df['Regressor'] = regressor
                        dfs.append(df)
                    except Exception as e:
                        print(f"Failed to read {fpath}: {e}")

    if not dfs:
        raise RuntimeError("No result files were loaded. Check paths or run regressors first.")

    combined_df = pd.concat(dfs, ignore_index=True)
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

        # prepares data for all three metrics
        heatmap_data_dict = {}
        for metric in metrics:
            pivot = df_model.pivot(index="Dataset", columns="Regressor", values=metric)

            # clips the values to a reasonable range for visualization
            if metric == "R2":
                pivot = pivot.clip(lower=-1, upper=1)

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
            
            heatmap_data_dict[metric] = heatmap_data.astype(float)

        # creates combined plot with all three metrics side by side
        if len(heatmap_data_dict) == 3:  # combined plot only, if all metrics are available
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                heatmap_data = heatmap_data_dict[metric]
                
                sns.heatmap(heatmap_data, cmap="coolwarm", center=0, annot=True, fmt=".3f", annot_kws={"size": 12}, linewidths=0.5, square=True, cbar=True, ax=ax)
                ax.set_title(f"{metric} Avg Difference", fontsize=14, fontweight='bold')
                ax.set_xlabel("Regressor B", fontweight='bold', fontsize=12)

                if i == 0:
                    ax.set_ylabel("Regressor A", fontweight='bold', fontsize=12)
                else:
                    ax.set_ylabel("")
                
                ax.tick_params(axis='both', which='major', labelsize=11)
            
            fig.suptitle(f"{model_arch} - Regressor Performance Comparison\nRed = A better, Blue = B better", fontsize=16, fontweight='bold', y=1.05)
            
            plt.subplots_adjust(wspace=0.15)
            plt.tight_layout()

            # saves combined plot
            filename = "combined_summary_heatmap.pdf"
            plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight", facecolor='white', edgecolor='none')
            plt.close()
            print(f"Saved combined summary heatmap to: {os.path.join(out_dir, filename)}")
        
        # saves individual plots
        for metric in metrics:
            if metric in heatmap_data_dict:
                heatmap_data = heatmap_data_dict[metric]
                
                plt.figure(figsize=(8, 6))
                ax = sns.heatmap(heatmap_data, cmap="coolwarm", center=0, annot=True, fmt=".3f", annot_kws={"size": 10}, linewidths=0.5, square=True, cbar=True)
                ax.collections[0].colorbar.set_label(f"{metric} Difference (A - B)", fontsize=10)
                plt.title(f"{model_arch} - {metric} Avg Difference (Regressor vs. Regressor)", fontsize=12, fontweight='bold')
                plt.suptitle("Red = A better, Blue = B better", fontsize=9, y=0.98)
                plt.xlabel("Regressor B", fontweight='bold')
                plt.ylabel("Regressor A", fontweight='bold')
                plt.tight_layout()

                filename = f"{metric}_summary_heatmap.pdf"
                plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight", facecolor='white', edgecolor='none')
                plt.close()
                print(f"Saved individual summary heatmap to: {os.path.join(out_dir, filename)}")


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

                            if metric == "R2":
                                val_a = max(min(val_a, 1), -1)
                                val_b = max(min(val_b, 1), -1)

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
                plt.title(f"{model_arch} - {dataset} - {metric} (Regressor A - B)", fontsize=12, fontweight='bold')
                plt.suptitle("Red = A better, Blue = B better", fontsize=9, y=0.98)
                plt.xlabel("Regressor B", fontweight='bold')
                plt.ylabel("Regressor A", fontweight='bold')
                plt.tight_layout()
                filename = f"{metric}_pairwise_heatmap.pdf"
                plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight", facecolor='white', edgecolor='none')
                plt.close()
                print(f"Saved pairwise heatmap to: {os.path.join(out_dir, filename)}")


if __name__ == "__main__":
    print("Loading and combining results")
    combined_df = load_and_combine_results()
    print("Generating heatmaps")
    generate_pairwise_summary_heatmaps(combined_df)
    print("Generating per-dataset pairwise heatmaps")
    generate_dataset_pairwise_heatmaps(combined_df)
