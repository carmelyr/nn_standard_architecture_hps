import os
import json
import numpy as np
import pandas as pd
import torch
import time
import shutil
import random
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from config_spaces import get_transformer_config_space, transformer_seed
from model_builder import build_transformer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from sktime.datasets import load_from_tsfile
from sklearn.preprocessing import LabelEncoder
from scipy.io import arff
import argparse

datasets = {
    #"classification_ozone": "datasets/classification_ozone/X_train.csv",
    #"AbnormalHeartbeat": "datasets/AbnormalHeartbeat/AbnormalHeartbeat_TRAIN.txt",
    #"Adiac": "datasets/Adiac/Adiac_TRAIN.txt",
    #"ArrowHead": "datasets/ArrowHead/ArrowHead_TRAIN.txt",
    #"Beef": "datasets/Beef/Beef_TRAIN.txt",
    #"BeetleFly": "datasets/BeetleFly/BeetleFly_TRAIN.txt",
    #"BirdChicken": "datasets/BirdChicken/BirdChicken_TRAIN.txt",
    #"BinaryHeartbeat": "datasets/BinaryHeartbeat/BinaryHeartbeat_TRAIN.txt",
    #"Car": "datasets/Car/Car_TRAIN.txt",
    #"CBF": "datasets/CBF/CBF_TRAIN.txt",
    #"CatsDogs": "datasets/CatsDogs/CatsDogs_TRAIN.ts",
    #"ChlorineConcentration": "datasets/ChlorineConcentration/ChlorineConcentration_TRAIN.txt",
    #"CinCECGTorso": "datasets/CinCECGTorso/CinCECGTorso_TRAIN.txt",
    #"CounterMovementJump": "datasets/CounterMovementJump/CounterMovementJump_TRAIN.ts",
    #"DucksAndGeese": "datasets/DucksAndGeese/DucksAndGeese_TRAIN.ts",
    #"EigenWorms": "datasets/EigenWorms/EigenWorms_TRAIN.ts",
    #"FiftyWords": "datasets/FiftyWords/FiftyWords_TRAIN.txt",
    #"FaultDetectionB": "datasets/FaultDetectionB/FaultDetectionB_TRAIN.ts",
    "HouseTwenty": "datasets/HouseTwenty/HouseTwenty_TRAIN.txt",
    #"KeplerLightCurves": "datasets/KeplerLightCurves/KeplerLightCurves_TRAIN.ts",
    #"RightWhaleCalls": "datasets/RightWhaleCalls/RightWhaleCalls_TRAIN.arff",
}

class JSONLogger(pl.callbacks.Callback):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)
        train_acc = trainer.callback_metrics.get("train_accuracy", None)
        self.metrics["train_loss"].append(train_loss.item() if train_loss is not None else None)
        self.metrics["train_accuracy"].append(train_acc.item() if train_acc is not None else None)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss", None)
        val_acc = trainer.callback_metrics.get("val_accuracy", None)
        self.metrics["val_loss"].append(val_loss.item() if val_loss is not None else None)
        self.metrics["val_accuracy"].append(val_acc.item() if val_acc is not None else None)

class EpochTimeLogger(pl.callbacks.Callback):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is not None:
            duration = time.time() - self.epoch_start_time
            if "epoch_times" not in self.metrics:
                self.metrics["epoch_times"] = []
            self.metrics["epoch_times"].append(duration)

def load_dataset(path, dataset_name=None):
    if dataset_name == "classification_ozone":
        X_train = pd.read_csv("datasets/classification_ozone/X_train.csv").values
        y_train = pd.read_csv("datasets/classification_ozone/y_train.csv").values.squeeze().astype(int)
        X_test = pd.read_csv("datasets/classification_ozone/X_test.csv").values
        y_test = pd.read_csv("datasets/classification_ozone/y_test.csv").values.squeeze().astype(int)

    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        y_train = df.iloc[:, 0].values.astype(int)
        X_train = df.iloc[:, 1:].values
        X_test, y_test = X_train, y_train

    elif path.endswith(".txt"):
        train_data = np.loadtxt(path)
        y_train = train_data[:, 0].astype(int)
        X_train = train_data[:, 1:]

        test_path = path.replace("_TRAIN.txt", "_TEST.txt")
        test_data = np.loadtxt(test_path)
        y_test = test_data[:, 0].astype(int)
        X_test = test_data[:, 1:]

    elif path.endswith(".ts"):
        def parse_multivariate_ts_file(filepath):
            data = []
            labels = []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("@") or line.startswith("#"):
                        continue
                    segments = line.split(":")
                    series = []
                    for segment in segments[:-1]:
                        values = list(map(float, segment.split(",")))
                        series.append(values)
                    data.append(np.array(series))    # (channels, time)
                    labels.append(segments[-1])
            X = np.stack(data)                      # (samples, channels, time)
            y = np.array(labels)
            return X.transpose(0, 2, 1), y          # (samples, time, channels)

        train_path = path.replace("_TEST.ts", "_TRAIN.ts")
        test_path = path

        X_train, y_train = parse_multivariate_ts_file(train_path)
        X_test, y_test = parse_multivariate_ts_file(test_path)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    elif path.endswith(".arff"):
        train_path = path.replace("_TEST.arff", "_TRAIN.arff")
        test_path = path

        def load_arff(path):
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)

            if isinstance(df[df.columns[-1]].iloc[0], bytes):
                df[df.columns[-1]] = df[df.columns[-1]].str.decode("utf-8")
            
            y = df[df.columns[-1]].values
            series_column = df[df.columns[0]].values

            X = []
            for row in series_column:
                if isinstance(row, (np.ndarray, np.float64, float)):
                    values = np.array([float(row)])
                elif isinstance(row, (str, bytes)):
                    if isinstance(row, bytes):
                        row = row.decode("utf-8")
                    values = np.array([float(x) for x in row.strip().split(",")])
                else:
                    raise ValueError(f"Unexpected ARFF type: {type(row)} with value {row}")
                X.append(values)

            X = np.array(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1, 1)
            elif X.ndim == 2:
                X = X[:, :, np.newaxis]
            return X, y

        X_train, y_train = load_arff(train_path)
        X_test, y_test = load_arff(test_path)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    if np.min(y) == 1:
        y -= 1

    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    X = np.where(np.isposinf(X), 1e6, X)
    X = np.where(np.isneginf(X), -1e6, X)

    TARGET_SEQ_LEN = 1200 
    
    if X.shape[1] > TARGET_SEQ_LEN:
        original_length = X.shape[1]
        downsample_factor = int(np.ceil(original_length / TARGET_SEQ_LEN))
        print(f"[INFO] Downsampling sequence length from {original_length} to {original_length // downsample_factor} (factor: {downsample_factor}x)")
        
        if X.ndim == 3:
            X = X[:, ::downsample_factor, :]
        else: 
            X = X[:, ::downsample_factor]

    if X.ndim == 1:
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    elif X.ndim == 2:
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    
    y_tensor = torch.tensor(y, dtype=torch.long)

    num_classes = len(np.unique(y))
    dataset = TensorDataset(X_tensor, y_tensor)
    dataset.targets = y  # For stratified split

    print(f"\nDataset: {dataset_name}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes: {np.unique(y)}, counts: {np.bincount(y)}")
    print(f"NaN in X: {np.isnan(X).sum()}, Inf in X: {np.isinf(X).sum()}")
    
    return dataset, X_tensor.shape[1:], num_classes

# this method is used to save the results of the training
def save_results(arch_name, dataset_name, config_idx, metrics, transformer_seed):
    # directory structure
    config_id = config_idx + 1
    result_dir = os.path.join("results", arch_name, dataset_name, f"config_{config_id}")
    os.makedirs(result_dir, exist_ok=True)

    # full filename
    filename = f"{dataset_name}_config_{config_id}_seed{transformer_seed}.json"
    out_path = os.path.join(result_dir, filename)

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✔ Saved: {out_path}")

# this method is used to train the transformer model
def train_transformer(selected_config_indices):
    seeds = [2]
    config_space = get_transformer_config_space()

    for transformer_seed in seeds:
        if os.path.exists('lightning_logs'):
            shutil.rmtree('lightning_logs')
        if os.path.exists('__pycache__'):
            shutil.rmtree('__pycache__')
        print(f"\n=== Running for SEED {transformer_seed} ===")

        # all global seeds
        random.seed(transformer_seed)
        np.random.seed(transformer_seed)
        torch.manual_seed(transformer_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(transformer_seed)

        if torch.cuda.is_available():
            accelerator = "gpu"
            print("Using CUDA backend")
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            print("Using MPS backend")
        else:
            accelerator = "cpu"
            print("Using CPU")

        # 100 configurations
        for config_idx in selected_config_indices:
            config_id = config_idx + 1

            for dataset_name, dataset_path in datasets.items():
                try:
                    if transformer_seed == 2:
                        config_file = f"{dataset_name}_config_{config_id}_seed1.json"
                        config_path = os.path.join("results", "Transformer", dataset_name, f"config_{config_id}", config_file)

                        if os.path.exists(config_path):
                            with open(config_path, "r") as f:
                                sampled_config = json.load(f)["hyperparameters"]
                            print(f"[INFO] Loaded hyperparameters from seed1 for {dataset_name} config_{config_id}")
                        else:
                            print(f"[SKIP] Missing seed1 config for {dataset_name} config_{config_id}")
                            continue            # skips the dataset if seed1 config is missing
                    else:
                        sampled_config = dict(config_space.sample_configuration())
                        sampled_config = {k: v.item() if isinstance(v, np.generic) else v for k, v in sampled_config.items()}
                        
                    print(f"\nTraining on {dataset_name} with config {config_idx + 1}")
                    
                    dataset, input_shape, num_classes = load_dataset(dataset_path, dataset_name)
                    print(f"Input shape: {input_shape}, Num classes: {num_classes}")

                    y_numpy = dataset.targets
                    if len(y_numpy) > 1000:
                        train_size = 50 * num_classes
                        sss = StratifiedShuffleSplit(n_splits=5, train_size=train_size, random_state=transformer_seed)
                    else:
                        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=transformer_seed)

                    fold_metrics = []

                    for fold_idx, (train_idx, val_idx) in enumerate(sss.split(np.zeros(len(y_numpy)), y_numpy)):
                        print(f"\nFold {fold_idx + 1}")

                        train_dataset = Subset(dataset, train_idx)
                        val_dataset = Subset(dataset, val_idx)

                        seq_len = input_shape[0] if isinstance(input_shape, (tuple, list)) else 0
                        if seq_len > 50000:
                            batch_size = 8
                        elif seq_len > 20000:
                            batch_size = 16
                        elif seq_len > 10000:
                            batch_size = 32
                        else:
                            batch_size = 32

                        train_loader = DataLoader(
                            train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            drop_last=(len(train_dataset) > batch_size),
                            num_workers=0
                        )
                        val_loader = DataLoader(
                            val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=0
                        )

                        model = build_transformer(sampled_config, input_shape, num_classes)

                        fold_log = {
                            "train_loss": [],
                            "val_loss": [],
                            "train_accuracy": [],
                            "val_accuracy": [],
                            "epoch_times": []
                        }

                        early_stopping = EarlyStopping(
                            monitor="val_loss", 
                            patience=30, 
                            mode="min",
                            min_delta=0.001
                        )
                        trainer = pl.Trainer(
                            accelerator="cuda",
                            precision="16-mixed",
                            max_epochs=1024,
                            enable_checkpointing=False,
                            logger=False,
                            callbacks=[
                                JSONLogger(fold_log),
                                EpochTimeLogger(fold_log),
                                early_stopping
                            ],
                            gradient_clip_val=1.0,
                            gradient_clip_algorithm="norm",
                            accumulate_grad_batches=1,
                            log_every_n_steps=1
                        )

                        trainer.fit(model, train_loader, val_loader)
                        fold_log["epochs"] = trainer.current_epoch
                        fold_metrics.append(fold_log)

                        torch.cuda.empty_cache()
                        shutil.rmtree("lightning_logs", ignore_errors=True)
                        shutil.rmtree("__pycache__", ignore_errors=True)


                    avg_metrics = {
                        "train_loss": np.mean([m["train_loss"][-1] for m in fold_metrics]),
                        "val_loss": np.mean([m["val_loss"][-1] for m in fold_metrics]),
                        "train_accuracy": np.mean([m["train_accuracy"][-1] for m in fold_metrics]),
                        "val_accuracy": np.mean([m["val_accuracy"][-1] for m in fold_metrics]),
                        "epoch_times": np.mean([np.mean(m["epoch_times"]) for m in fold_metrics]),
                        "epochs": np.mean([m["epochs"] for m in fold_metrics]),
                        "folds": 5
                    }

                    metrics = {
                        "hyperparameters": sampled_config,
                        "dataset_stats": {
                            "name": dataset_name,
                            "input_shape": input_shape,
                            "num_classes": num_classes,
                            "folds": 5,
                            "train_size": int(np.mean([len(train_idx) for train_idx, _ in sss.split(np.zeros(len(y_numpy)), y_numpy)])),
                            "val_size": int(np.mean([len(val_idx) for _, val_idx in sss.split(np.zeros(len(y_numpy)), y_numpy)])),
                        },
                        "fold_logs": fold_metrics,
                        "epochs": [m["epochs"] for m in fold_metrics],
                        "train_loss": [m["train_loss"][-1] for m in fold_metrics],
                        "val_loss": [m["val_loss"][-1] for m in fold_metrics],
                        "train_accuracy": [m["train_accuracy"][-1] for m in fold_metrics],
                        "val_accuracy": [m["val_accuracy"][-1] for m in fold_metrics],
                        "epoch_times": [np.mean(m["epoch_times"]) for m in fold_metrics],
                        "avg_train_loss": avg_metrics["train_loss"],
                        "avg_val_loss": avg_metrics["val_loss"],
                        "avg_train_accuracy": avg_metrics["train_accuracy"],
                        "avg_val_accuracy": avg_metrics["val_accuracy"],
                        "avg_epoch_time": avg_metrics["epoch_times"],
                        "avg_epochs": avg_metrics["epochs"]
                    }

                    save_results("Transformer", dataset_name, config_idx, metrics, transformer_seed)

                    torch.cuda.empty_cache()
                    shutil.rmtree("lightning_logs", ignore_errors=True)
                    shutil.rmtree("__pycache__", ignore_errors=True)

                except Exception as e:
                    print(f"Failed for {dataset_name} config {config_idx}: {e}")
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, help="Comma-separated list of config indices to run, e.g., 2,5,8")
    args = parser.parse_args()

    if args.configs:
        selected_config_ids = [int(i) for i in args.configs.split(",")]
        selected_indices = [i - 1 for i in selected_config_ids]  # shift back to 0-indexed
    else:
        selected_indices = list(range(100))  # default: run all

    train_transformer(selected_indices)

