"""
Feature Importance Heatmap Generator

This script creates comprehensive heatmaps that show the most important features 
(hyperparameters) for each neural network model and dataset combination across 
different surrogate models (Random Forest, XGBoost, Linear Regression).

The heatmaps helps to identify:
1. Which hyperparameters are most influential for each model architecture
2. How feature importance patterns vary across datasets
3. Differences in feature prioritization between surrogate models
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from collections import defaultdict
import warnings
import glob
import matplotlib.colors as mcolors
warnings.filterwarnings('ignore')

SURROGATE_DATA_DIR = "surrogate_datasets"
OUTPUT_DIR = "feature_importance_heatmaps"
MODELS = ["CNN", "FCNN", "GRU", "LSTM", "Transformer"]
RESULTS_DIR = "results"

REGRESSORS = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, verbosity=0, random_state=42),
    'Linear Regression': LinearRegression()
}

REGRESSOR_COLORS = {
    'Random Forest': 'Greens',
    'XGBoost': 'Reds',
    'Linear Regression': 'Blues'
}

# this function extracts feature importances from the trained model
# X: Feature matrix
# y: Target values
# returns: Dictionary of feature importances
def get_feature_importances(regressor_name, model, X, y):
    if regressor_name in ['Random Forest', 'XGBoost']:
        # tree-based models have built-in feature_importances_
        importances = model.feature_importances_
    elif regressor_name == 'Linear Regression':
        # absolute value of coefficients
        importances = np.abs(model.coef_)

        # normalizes to sum = 1
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance

    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")
    
    # creates a dictionary of feature importances
    feature_importance_dict = {}
    for i, feature in enumerate(X.columns):
        feature_importance_dict[feature] = importances[i]
    
    return feature_importance_dict

# this function loads and processes a dataset for feature importance analysis
# returns: Tuple of (X, y, feature_names) or None if dataset cannot be processed
def load_and_process_dataset(model_name, dataset_name):
    dataset_path = os.path.join(SURROGATE_DATA_DIR, model_name, dataset_name, f"{model_name}_epochs_1-20.csv")
    
    if not os.path.exists(dataset_path):
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        
        if len(df) < 10:
            return None
        
        # checks target variance before proceeding
        y_temp = df["val_acc_20"].dropna()
        if len(y_temp) < 5 or y_temp.std() < 0.001:
            # if target has very low variance, adds small noise to help models
            y_temp = y_temp + np.random.normal(0, 0.001, len(y_temp))
            
        if "val_acc_20" not in df.columns:
            return None
        
        HP_PREFIXES = (
            "dropout", "learning_rate", "num_", "activation",
            "kernel", "pooling", "bidirectional", "weight_decay", "ff_dim"
        )

        HP_EXACT = {"hidden_units", "output_activation", "pooling_size", "kernel_size"}

        hp_cols = [col for col in df.columns if col.startswith(HP_PREFIXES) or col in HP_EXACT]

        if not hp_cols:
            return None
        
        X = df[hp_cols]
        y = df["val_acc_20"].dropna()
        
        # adds small noise if target has very low variance
        if y.std() < 0.001:
            y = y + np.random.normal(0, 0.001, len(y))
            
        X = X.loc[y.index]
        
        # handles categorical variables
        X = pd.get_dummies(X)

        # imputes missing values
        X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X),  columns=X.columns)

        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        
        if len(X) < 10:
            return None
            
        return X, y, list(X.columns)
        
    except Exception as e:
        print(f"Error processing {model_name}/{dataset_name}: {e}")
        return None

# this function extracts feature importances for all model-dataset-regressor combinations
# returns: Nested dictionary of feature importances
def extract_all_feature_importances():
    all_importances = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for model_name in MODELS:
        model_path = os.path.join(SURROGATE_DATA_DIR, model_name)
        if not os.path.isdir(model_path):
            continue

        datasets = []
        if os.path.isdir(model_path):
            for subdir in os.listdir(model_path):
                subdir_path = os.path.join(model_path, subdir)
                if os.path.isdir(subdir_path) and subdir != '.DS_Store':
                    csv_file = os.path.join(subdir_path, f"{model_name}_epochs_1-20.csv")
                    if os.path.exists(csv_file):
                        datasets.append(subdir)
        
        for dataset_name in datasets:
            print(f"  Processing dataset: {dataset_name}")

            # loads and preprocesses data
            result = load_and_process_dataset(model_name, dataset_name)
            if result is None:
                continue
                
            X, y, feature_names = result
            
            # tests each regressor
            for regressor_name, regressor in REGRESSORS.items():
                try:
                    # adaptive train/test split based on dataset size
                    if len(X) < 20:
                        test_size = 0.3
                    else:
                        test_size = 0.2
                    
                    min_test_samples = max(2, int(len(X) * 0.1))
                    actual_test_size = max(test_size, min_test_samples / len(X))
                    
                    # splits data for training
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=actual_test_size, random_state=42, stratify=None)

                    # trains surrogate model
                    regressor.fit(X_train, y_train)
                    
                    y_pred = regressor.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > -0.5:
                        importances = get_feature_importances(regressor_name, regressor, X, y)
                        all_importances[regressor_name][model_name][dataset_name] = importances
                        
                    else:
                        # fallback: Use correlation-based importance when model fails
                        print(f"    Using correlation fallback for {regressor_name} (R²={r2:.4f})")
                        correlation_importances = {}
                        for col in X.columns:
                            corr = abs(X[col].corr(y))
                            if not pd.isna(corr):
                                correlation_importances[col] = corr
                        
                        if correlation_importances:
                            # normalizes correlations
                            total_corr = sum(correlation_importances.values())
                            if total_corr > 0:
                                correlation_importances = {k: v/total_corr for k, v in correlation_importances.items()}
                                all_importances[regressor_name][model_name][dataset_name] = correlation_importances
                        
                except Exception as e:
                    print(f"    Error with {regressor_name}: {e}")
                    continue
    
    return all_importances

# this function builds full importance DataFrames for a specific regressor
# returns: dict of DataFrames, where each DF has rows = datasets, cols = ALL features (union across datasets), values = normalized importance
def build_full_importance_frames(all_importances, regressor_name):
    model_frames = {}
    if regressor_name not in all_importances:
        return model_frames

    for model_name, ds_dict in all_importances[regressor_name].items():
        # collects union of features
        all_feats = set()
        for importances in ds_dict.values():
            all_feats.update(importances.keys())
        all_feats = sorted(all_feats)

        # builds rows per dataset
        rows = []
        idx = []
        for dataset_name, importances in ds_dict.items():
            vec = [importances.get(f, 0.0) for f in all_feats]
            s = sum(vec)
            if s > 0:
                vec = [v / s for v in vec]  # normalizes row to sum 1
            rows.append(vec)
            idx.append(dataset_name)

        if rows:
            df = pd.DataFrame(rows, index=idx, columns=all_feats)
            model_frames[model_name] = df.sort_index()
    return model_frames

# this function creates heatmaps for all features, where rows = datasets, columns = features, values = normalized importance
def plot_all_features_heatmap(model_frames, regressor_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name, df in model_frames.items():
        if df.empty:
            continue

        # orders columns (groups by semantic prefixes)
        ordered_cols = sorted(df.columns, key=lambda c: (
            0 if 'learning_rate' in c else
            1 if 'num_' in c or c in ['hidden_units','ff_dim','num_heads'] else
            2 if 'kernel_size' in c or 'pooling' in c else
            3 if 'dropout' in c or 'weight_decay' in c else
            4 if 'activation' in c else 5
        , c))
        df = df[ordered_cols]

        # --- special case: compact labels for output_activation_* ---
        def _pretty_col(c):
            if isinstance(c, str) and c.startswith('output_activation_'):
                val = c.split('_', 2)[-1]
                return f"out. act.: {val}"
            return c

        df = df.rename(columns={c: _pretty_col(c) for c in df.columns})

        # figure size scales with number of features/datasets
        h = max(6, 0.4 * len(df.index) + 2)
        w = max(8, 0.25 * len(df.columns) + 3)
        plt.figure(figsize=(w, h))
        # chooses color palette by regressor
        cmap_map = {
            "Linear Regression": "Blues",
            "Random Forest": "Greens",
            "XGBoost": "Reds"
        }
        chosen_cmap = cmap_map.get(regressor_name, "mako")

        ax = sns.heatmap(df, cmap=chosen_cmap, vmin=0.0, vmax=1.0, linewidths=0.3)

        ax.set_title(f'All Hyperparameter Importances — {regressor_name} — {model_name}', fontsize=14, fontweight='bold', pad=14)
        ax.set_xlabel('Hyperparameters', fontsize=12)
        ax.set_ylabel('Datasets', fontsize=12)
        plt.xticks(rotation=60, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f'fi_all_features_{regressor_name.replace(" ","_")}_{model_name}.pdf')
        plt.savefig(out, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {out}")

# this function tries to extract validation accuracy from a JSON object
# returns a float in [0,1] if found, else None.
def _safe_get_val_acc(obj):
    preferred_keys = [
        "final_val_accuracy", "val_accuracy", "val_acc",
        "val_acc_final", "best_val_acc", "best_val_accuracy",
        "valid_accuracy", "validation_accuracy",
    ]

    # direct hit
    for k in preferred_keys:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], (int, float)):
            v = float(obj[k])
            # normalizes if looks like percentage
            return v/100.0 if v > 1.0 else v

    # sometimes nested dicts/lists
    def _walk(o):
        if isinstance(o, dict):
            # checks preferred keys at this level first
            for k in preferred_keys:
                if k in o and isinstance(o[k], (int, float)):
                    v = float(o[k])
                    return v/100.0 if v > 1.0 else v
            # heuristic: any key containing both 'val' and 'acc'
            for k, v in o.items():
                if isinstance(v, (int, float)) and ('val' in k.lower()) and ('acc' in k.lower()):
                    vv = float(v)
                    return vv/100.0 if vv > 1.0 else vv
            # recurses
            for v in o.values():
                r = _walk(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for item in o:
                r = _walk(item)
                if r is not None:
                    return r
        return None

    return _walk(obj)

# this function loads config performance data from a JSON file
def load_config_performance_from_json(dataset_name, models=MODELS, use_max_over_epochs=False, smooth_window=5):
    """
    Aggregates performance per CONFIG (not per run), to match representative curves:
      - collects all runs (seeds/folds) for a config
      - pads & averages per-epoch across runs -> config_mean
      - smooths (centered rolling mean) like representative curves
      - takes FINAL (last valid) or MAX over epochs
    Returns DataFrame with columns: ['Model', 'Config', 'Score'].
    """
    recs = []
    for m in models:
        pattern = os.path.join(RESULTS_DIR, m, dataset_name, "config_*", f"{dataset_name}_config_*_seed*.json")

        # groups run files by config id
        by_config = {}
        for fp in glob.glob(pattern):
            cfg = os.path.basename(os.path.dirname(fp))  # "config_XX"
            by_config.setdefault(cfg, []).append(fp)

        for cfg, files in by_config.items():
            seed_curves = []
            for fp in files:
                try:
                    with open(fp, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                # extracts per-epoch validation accuracy curves
                if "fold_logs" in data:
                    for fold in data["fold_logs"]:
                        vals = fold.get("val_accuracy", [])
                        seed_curves.append([v if v is not None else np.nan for v in vals])
                else:
                    vals = data.get("val_accuracy", [])
                    seed_curves.append([v if v is not None else np.nan for v in vals])

            if not seed_curves:
                continue

            # pads to same length and averages across runs (seeds x folds)
            max_len = max(len(c) for c in seed_curves)
            padded = np.full((len(seed_curves), max_len), np.nan)
            for i, c in enumerate(seed_curves):
                padded[i, :len(c)] = c
            config_mean = np.nanmean(padded, axis=0)

            # smooths like representative curves (centered rolling mean)
            if smooth_window and smooth_window > 1:
                series = pd.Series(config_mean)
                config_mean = series.rolling(smooth_window, min_periods=1, center=True).mean().to_numpy()

            # chooses final (last valid) or max across epochs
            if use_max_over_epochs:
                score = float(np.nanmax(config_mean))
            else:
                score = next((v for v in reversed(config_mean) if not np.isnan(v)), np.nan)

            if not np.isnan(score):
                score = max(0.0, min(1.0, float(score)))
                recs.append({"Model": m, "Config": cfg, "Score": score})

    return pd.DataFrame(recs)

# this function discovers datasets
def discover_datasets(results_dir=RESULTS_DIR, models=MODELS):
    ds = set()
    for m in models:
        base = os.path.join(results_dir, m)
        if os.path.isdir(base):
            for name in os.listdir(base):
                path = os.path.join(base, name)
                if os.path.isdir(path):
                    ds.add(name)
    return sorted(ds)

# this function creates a 5-boxplot figure (one per model) for performance distribution
def boxplot_performance_distribution(dataset_name, use_max_over_epochs=False, smooth_window=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_config_performance_from_json(dataset_name, use_max_over_epochs=use_max_over_epochs, smooth_window=smooth_window)
    
    if df.empty:
        print(f"No performance data found for dataset: {dataset_name}")
        return

    plt.figure(figsize=(8, 5))
    order = [m for m in MODELS if m in df["Model"].unique()]

    # boxplot
    ax = sns.boxplot(
        data=df, x="Model", y="Score", order=order,
        color='#f8bbd9',
        boxprops=dict(facecolor='#f8bbd9', edgecolor='#e91e63', linewidth=1.5),
        whiskerprops=dict(color='#e91e63', linewidth=1.5),
        capprops=dict(color='#e91e63', linewidth=1.5),
        medianprops=dict(color='#ad1457', linewidth=2),
        flierprops=dict(markerfacecolor='#ec407a', markeredgecolor='#e91e63',
                        markersize=6, alpha=0.7)
    )

    ax.set_title(f'Performance Distribution across Configurations — {dataset_name}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Model')
    ylabel = "Max Validation Accuracy (per config)" if use_max_over_epochs else "Final Validation Accuracy (per config)"
    ax.set_ylabel(ylabel)
    plt.ylim(0.0, 1.01)             # small headroom to show 1.0 clearly
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f'perf_boxplots_{dataset_name}.pdf')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")

# this function retrieves the top feature for each model-dataset combination
# all_importances: Feature importance data
# returns: DataFrames with top features and their importance
def get_top_feature_per_combination(all_importances, regressor_name):
    combinations = []
    
    for model_name in MODELS:
        if model_name not in all_importances[regressor_name]:
            continue
            
        for dataset_name in all_importances[regressor_name][model_name]:
            importances = all_importances[regressor_name][model_name][dataset_name]
            
            if importances:
                # finds the feature with highest importance
                top_feature = max(importances.items(), key=lambda x: x[1])[0]
                top_importance = max(importances.values())
                
                combinations.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Top_Feature': top_feature,
                    'Importance': top_importance
                })
    
    if not combinations:
        return None
        
    df = pd.DataFrame(combinations)
    
    # creates pivot table with top features
    pivot_features = df.pivot(index='Dataset', columns='Model', values='Top_Feature')
    pivot_importance = df.pivot(index='Dataset', columns='Model', values='Importance')
    
    return pivot_features, pivot_importance

# this function creates a heatmap for feature importance
# pivot_features: DataFrame with top features
# pivot_importance: DataFrame with importance values
# regressor_name: Name of the regressor
def create_feature_importance_heatmap(pivot_features, pivot_importance, regressor_name):
    if pivot_features is None or pivot_importance is None:
        print(f"No data available for {regressor_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, len(pivot_features) * 0.5 + 3))
    
    # Plot 1: Feature names heatmap
    feature_matrix = pivot_features.copy()

    feature_categories = {
        'learning_rate': 1,
        'dropout': 2,
        'num_layers': 3,
        'num_filters': 4,
        'hidden_units': 5,
        'activation': 6,
        'output_activation': 6,
        'kernel_size': 7,
        'pooling': 8,
        'pooling_size': 8,
        'weight_decay': 9,
        'bidirectional': 10,
        'num_heads': 11,
        'ff_dim': 12,
        # additional features (if needed)
        'optimizer': 13,
        'batch_size': 14,
        'embedding': 15,
        'dense': 16,
    }

    pink_colors = [
        '#ffffff',  # 0 - white for missing values
        '#fce4ec',  # 1 - very light pink
        '#f8bbd9',  # 2 - light pink
        '#f48fb1',  # 3 - medium light pink
        '#f06292',  # 4 - medium pink
        '#ec407a',  # 5 - medium dark pink
        '#e91e63',  # 6 - dark pink
        '#d81b60',  # 7 - darker pink
        '#c2185b',  # 8 - very dark pink
        '#ad1457',  # 9 - deep pink
        '#880e4f',  # 10 - deepest pink
        '#ff4081',  # 11 - accent pink
        '#f50057',  # 12 - bright pink
        '#e1bee7',  # 13 - lavender pink
        '#ce93d8',  # 14 - purple pink
        '#ba68c8',  # 15 - violet pink
        '#ab47bc'   # 16 - purple
    ]
    
    while len(pink_colors) < max(feature_categories.values()) + 1:
        pink_colors.append('#ff69b4') # for fallback

    custom_pink_cmap = mcolors.ListedColormap(pink_colors)
    
    color_matrix = np.zeros(feature_matrix.shape)
    annotations = feature_matrix.copy()
    
    for i in range(feature_matrix.shape[0]):
        for j in range(feature_matrix.shape[1]):
            feature = feature_matrix.iloc[i, j]
            if pd.notna(feature):
                # categorizes feature
                category = 0
                for key, value in feature_categories.items():
                    if key in str(feature).lower():
                        category = value
                        break
                color_matrix[i, j] = category
                
                # formats feature name for display
                feature_str = str(feature)
                
                categorical_features = ['activation', 'output_activation', 'pooling']

                # --- special case: output_activation_* -> "out. act.: <value>" ---
                if feature_str.startswith('output_activation_'):
                    val = feature_str.split('_', 2)[-1]  # takes only the actual value
                    short_name = f"out. act.:\n{val}"

                else:
                    should_use_colon = False
                    for cat_feature in categorical_features:
                        if feature_str.startswith(cat_feature + '_') and not any(
                            feature_str.startswith(cat_feature + '_' + suffix) 
                            for suffix in ['size', 'rate', 'dim', 'layers', 'units', 'filters']):
                            should_use_colon = True
                            break

                    if should_use_colon:
                        parts = feature_str.split('_', 1)
                        if len(parts) > 1:
                            short_name = f"{parts[0]}:\n{parts[1]}"
                        else:
                            short_name = feature_str
                    else:
                        short_name = feature_str.replace('_', '\n')

                if len(short_name.replace('\n', '')) > 20:
                    lines = short_name.split('\n')
                    if len(lines) > 1:
                        if len(lines[1]) > 10:
                            lines[1] = lines[1][:8] + '..'
                        short_name = f"{lines[0]}\n{lines[1]}"
                    else:
                        short_name = short_name[:15] + '..'
                
                annotations.iloc[i, j] = short_name
            else:
                annotations.iloc[i, j] = ""
    
    # creates feature category heatmap
    sns.heatmap(
        color_matrix, 
        annot=annotations, 
        fmt='', 
        cmap=custom_pink_cmap, 
        ax=ax1,
        cbar=False,
        linewidths=0.5,
        xticklabels=pivot_features.columns,
        yticklabels=pivot_features.index,
        vmin=0,
        vmax=len(pink_colors)-1,
        annot_kws={'size': 16}
    )

    ax1.set_title(f'Most Important Features - {regressor_name}', fontweight='bold', fontsize=22, pad=20)
    ax1.set_xlabel('Model', fontweight='bold', fontsize=20)
    ax1.set_ylabel('Dataset', fontweight='bold', fontsize=20)
    ax1.tick_params(axis='x', rotation=45, labelsize=18)
    ax1.tick_params(axis='y', rotation=0, labelsize=18)
    
    # Plot 2: Importance values heatmap
    sns.heatmap(
        pivot_importance, 
        annot=True, 
        fmt='.3f', 
        cmap=REGRESSOR_COLORS[regressor_name],
        ax=ax2,
        cbar_kws={'label': 'Feature Importance', 'shrink': 0.8},
        linewidths=0.5,
        xticklabels=pivot_importance.columns,
        yticklabels=pivot_importance.index,
        annot_kws={'size': 16}
    )

    ax2.set_title(f'Feature Importance Values - {regressor_name}', fontweight='bold', fontsize=22, pad=20)
    ax2.set_xlabel('Model', fontweight='bold', fontsize=20)
    ax2.set_ylabel('Dataset', fontweight='bold', fontsize=20)
    ax2.tick_params(axis='x', rotation=45, labelsize=18)
    ax2.tick_params(axis='y', rotation=0, labelsize=18)
    
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Feature Importance', fontsize=18, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'feature_importance_heatmap_{regressor_name.replace(" ", "_")}.pdf')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved feature importance heatmap to: {output_path}")

# this function creates summary statistics about feature importance patterns
# all_importances: Feature importance data
def create_summary_statistics(all_importances):
    summary_data = []
    
    for regressor_name in REGRESSORS.keys():
        if regressor_name not in all_importances:
            continue
            
        regressor_data = all_importances[regressor_name]
        
        # counts occurrences of each feature as most important
        feature_counts = defaultdict(int)
        total_combinations = 0
        
        for model_name in regressor_data:
            for dataset_name in regressor_data[model_name]:
                importances = regressor_data[model_name][dataset_name]
                if importances:
                    top_feature = max(importances.items(), key=lambda x: x[1])[0]
                    feature_counts[top_feature] += 1
                    total_combinations += 1
        
        for feature, count in feature_counts.items():
            summary_data.append({
                'Regressor': regressor_name,
                'Feature': feature,
                'Count': count,
                'Percentage': (count / total_combinations) * 100 if total_combinations > 0 else 0
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(OUTPUT_DIR, 'feature_importance_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary statistics to: {summary_path}")
        
        print("\nTop 5 Most Important Features by Regressor:")
        for regressor_name in REGRESSORS.keys():
            regressor_summary = summary_df[summary_df['Regressor'] == regressor_name]
            if not regressor_summary.empty:
                top_features = regressor_summary.nlargest(5, 'Count')
                print(f"\n{regressor_name}:")
                for _, row in top_features.iterrows():
                    print(f"  {row['Feature']}: {row['Count']} times ({row['Percentage']:.1f}%)")

def main():
    print("=== Feature Importance Heatmap Generator ===")
    print("Extracting feature importances for all combinations:")

    sns.set_theme(context="notebook", style="whitegrid")

    all_importances = extract_all_feature_importances()

    if not all_importances:
        print("No feature importance data found!")
        return

    print(f"\nGenerating heatmaps for {len(REGRESSORS)} regressors...")

    # creates (existing) summary heatmaps for each regressor
    for regressor_name in REGRESSORS.keys():
        if regressor_name in all_importances:
            print(f"\nProcessing {regressor_name}...")
            result = get_top_feature_per_combination(all_importances, regressor_name)
            if result is not None:
                pivot_features, pivot_importance = result
                create_feature_importance_heatmap(pivot_features, pivot_importance, regressor_name)
            else:
                print(f"No valid data for {regressor_name}")

        # all-features heatmaps
        for regressor_name in REGRESSORS.keys():
            if regressor_name in all_importances:
                model_frames = build_full_importance_frames(all_importances, regressor_name)
                plot_all_features_heatmap(model_frames, regressor_name)

        print("\nGenerating performance distribution boxplots:")
        DATASETS_FOR_BOXPLOTS = discover_datasets()
        for ds in DATASETS_FOR_BOXPLOTS:
            boxplot_performance_distribution(ds, use_max_over_epochs=False, smooth_window=5)

        print("\nGenerating summary statistics:")
        create_summary_statistics(all_importances)

        print(f"\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
