import os
import json
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from config_spaces import get_fcnn_config_space, fcnn_seed
from model_builder import build_fcnn
from pytorch_lightning.callbacks import EarlyStopping

datasets = {
    "classification_ozone": "datasets/classification_ozone/X_train.csv",
    "Adiac": "datasets/Adiac/Adiac_TRAIN.txt",
    "ArrowHead": "datasets/ArrowHead/ArrowHead_TRAIN.txt",
    "Beef": "datasets/Beef/Beef_TRAIN.txt",
    "BeetleFly": "datasets/BeetleFly/BeetleFly_TRAIN.txt",
    "BirdChicken": "datasets/BirdChicken/BirdChicken_TRAIN.txt",
    "Car": "datasets/Car/Car_TRAIN.txt",
    "CBF": "datasets/CBF/CBF_TRAIN.txt",
    "ChlorineConcentration": "datasets/ChlorineConcentration/ChlorineConcentration_TRAIN.txt",
    "CinCECGTorso": "datasets/CinCECGTorso/CinCECGTorso_TRAIN.txt",
    "FiftyWords": "datasets/FiftyWords/FiftyWords_TRAIN.txt",
}

class JSONLogger(pl.callbacks.Callback):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    # this method is used to log metrics to a JSON file during training
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", torch.tensor(float('nan')))
        train_acc = trainer.callback_metrics.get("train_accuracy", torch.tensor(float('nan')))
        self.metrics["train_loss"].append(train_loss.item())
        self.metrics["train_accuracy"].append(train_acc.item())

    # this method is used to log metrics to a JSON file during validation
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float('nan')))
        val_acc = trainer.callback_metrics.get("val_accuracy", torch.tensor(float('nan')))
        self.metrics["val_loss"].append(val_loss.item())
        self.metrics["val_accuracy"].append(val_acc.item())

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
        X = pd.read_csv("datasets/classification_ozone/X_train.csv").values
        y = pd.read_csv("datasets/classification_ozone/y_train.csv").values.squeeze().astype(int)
        test_path = "datasets/classification_ozone/X_test.csv"
        y_test = pd.read_csv("datasets/classification_ozone/y_test.csv").values.squeeze().astype(int)

    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        y = df.iloc[:, 0].values.astype(int)
        X = df.iloc[:, 1:].values
        y_test = y

    elif path.endswith(".txt"):
        data = np.loadtxt(path)
        y = data[:, 0].astype(int)
        X = data[:, 1:]
        test_path = path.replace("_TRAIN.txt", "_TEST.txt")
        test_data = np.loadtxt(test_path)
        y_test = test_data[:, 0].astype(int)

    else:
        raise ValueError(f"Unsupported file format: {path}")

    print(f"{dataset_name}: unique labels BEFORE shift: {np.unique(y)}")

    # only shift labels if they start at 1
    if np.min(y) == 1:
        y -= 1
    if np.min(y_test) == 1:
        y_test -= 1

    # handles NaN and infinite values
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    X = np.where(np.isposinf(X), 1e6, X)
    X = np.where(np.isneginf(X), -1e6, X)

    if X.shape[0] < 100:
        def time_warp(x, factor=0.1):
            tt = np.arange(x.shape[0])
            tt_warped = tt + np.random.normal(0, factor * x.shape[0], size=x.shape[0])
            return np.interp(tt, tt_warped, x)
            
        def jitter(x, sigma=0.03):
            return x + np.random.normal(0, sigma, size=x.shape)
            
        def scale(x, sigma=0.1):
            return x * np.random.normal(1, sigma, size=x.shape)
        
        X_aug = []
        y_aug = []

        for _ in range(5):
            for x in X:
                x_warped = time_warp(x)
                x_jittered = jitter(x_warped)
                x_scaled = scale(x_jittered)
                X_aug.append(x_scaled)
            y_aug.extend(y.copy())
        
        X = np.concatenate([X] + [X_aug[i::len(X)] for i in range(5)])
        y = np.concatenate([y] + [y_aug[i::len(y)] for i in range(5)])

    if X.ndim == 2:
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    all_labels = np.concatenate([y, y_test])
    num_classes = len(np.unique(all_labels))

    dataset = TensorDataset(X_tensor, y_tensor)

    print(f"\nDataset: {dataset_name}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Classes: {np.unique(y)}, counts: {np.bincount(y)}")
    print(f"NaN in X: {np.isnan(X).sum()}, Inf in X: {np.isinf(X).sum()}")
    
    return dataset, X_tensor.shape[1:], num_classes

# this function saves the results of the training to a JSON file
"""def save_results(arch_name, dataset_name, config_idx, metrics, fcnn_):
    result_dir = os.path.join("results", arch_name)
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, f"{dataset_name}_config_{config_idx + 1}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved: {out_path}")"""
def save_results(arch_name, dataset_name, config_idx, metrics, fcnn_seed):
    # directory structure
    config_id = config_idx + 1
    result_dir = os.path.join("results", arch_name, dataset_name, f"config_{config_id}")
    os.makedirs(result_dir, exist_ok=True)

    # full filename
    filename = f"{dataset_name}_config_{config_id}_seed{fcnn_seed}.json"
    out_path = os.path.join(result_dir, filename)

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✔ Saved: {out_path}")


# this function trains the FCNN model on the datasets
def train_fcnn():
    config_space = get_fcnn_config_space()

    # 100 configurations
    for config_idx in range(100):
        sampled_config = dict(config_space.sample_configuration())
        sampled_config = {k: v.item() if isinstance(v, np.generic) else v for k, v in sampled_config.items()}

        for dataset_name, dataset_path in datasets.items():
            try:
                dataset, input_shape, num_classes = load_dataset(dataset_path, dataset_name)
                
                val_size = int(0.2 * len(dataset))
                train_size = len(dataset) - val_size
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                flat_input_size = int(np.prod(input_shape))
                model = build_fcnn(sampled_config, flat_input_size, num_classes)

                metrics = {
                    "hyperparameters": sampled_config,
                    "dataset_stats": {
                        "name": dataset_name,
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset),
                        "input_shape": input_shape,
                        "num_classes": num_classes
                    },
                    "epochs": [],
                    "train_loss": [],
                    "val_loss": [],
                    "train_accuracy": [],
                    "val_accuracy": []
                }

                early_stopping = EarlyStopping(monitor="val_loss", patience=30, mode="min")
                trainer = pl.Trainer(accelerator="cpu", max_epochs=1024, callbacks=[JSONLogger(metrics), EpochTimeLogger(metrics), early_stopping])

                trainer.fit(model, train_loader, val_loader)
                metrics["epochs"] = trainer.current_epoch
                save_results("FCNN", dataset_name, config_idx, metrics, fcnn_seed)

            except Exception as e:
                print(f"Failed for {dataset_name} config {config_idx}: {e}")
                continue

if __name__ == "__main__":
    train_fcnn()
