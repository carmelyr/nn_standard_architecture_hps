import os
import json
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from config_spaces import get_gru_config_space 
from model_builder import build_gru
from pytorch_lightning.callbacks import EarlyStopping

datasets = {
    #"classification_ozone": "datasets/classification_ozone/X_train.csv",
    #"Adiac": "datasets/Adiac/Adiac_TRAIN.txt",
    "ArrowHead": "datasets/ArrowHead/ArrowHead_TRAIN.txt",
    #"Beef": "datasets/Beef/Beef_TRAIN.txt",
    #"BeetleFly": "datasets/BeetleFly/BeetleFly_TRAIN.txt",
    #"BirdChicken": "datasets/BirdChicken/BirdChicken_TRAIN.txt",
    #"Car": "datasets/Car/Car_TRAIN.txt",
    #"CBF": "datasets/CBF/CBF_TRAIN.txt",
    #"ChlorineConcentration": "datasets/ChlorineConcentration/ChlorineConcentration_TRAIN.txt",
    #"CinCECGTorso": "datasets/CinCECGTorso/CinCECGTorso_TRAIN.txt",
    #"FiftyWords": "datasets/FiftyWords/FiftyWords_TRAIN.txt",
}

class JSONLogger(pl.callbacks.Callback):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)
        train_acc = trainer.callback_metrics.get("train_accuracy", None)
        self.metrics["train_loss"].append(train_loss.item() if train_loss else None)
        self.metrics["train_accuracy"].append(train_acc.item() if train_acc else None)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss", None)
        val_acc = trainer.callback_metrics.get("val_accuracy", None)
        self.metrics["val_loss"].append(val_loss.item() if val_loss else None)
        self.metrics["val_accuracy"].append(val_acc.item() if val_acc else None)

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

    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

    elif path.endswith(".txt"):
        data = np.loadtxt(path)
        y = data[:, 0].astype(int)
        X = data[:, 1:]

    # handles NaN and infinite values
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    X = np.where(np.isposinf(X), 1e6, X)
    X = np.where(np.isneginf(X), -1e6, X)

    label_mapping = {orig_label: idx for idx, orig_label in enumerate(sorted(np.unique(y)))}
    y = np.array([label_mapping[label] for label in y])
    print(f"[{dataset_name}] Label mapping: {label_mapping}")

    # data augmentation for small datasets
    if X.shape[0] < 100:
        def time_warp(x, factor=0.1):
            tt = np.arange(x.shape[0])
            tt_warped = tt + np.random.normal(0, factor*x.shape[0], size=x.shape[0])
            return np.interp(tt, tt_warped, x)
        X_aug = np.array([time_warp(x) for x in X])
        y_aug = y.copy()
        X = np.concatenate([X, X_aug])
        y = np.concatenate([y, y_aug])

    # converts the data to PyTorch tensors and handles the feature dimension
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(-1)
    
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset, X_tensor.shape[1:], len(label_mapping)

# this function is used to save the results of the training
def save_results(arch_name, dataset_name, config_idx, metrics):
    result_dir = os.path.join("results", arch_name)
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, f"{dataset_name}_config_{config_idx + 1}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved: {out_path}")

# this function is used to train the GRU model
def train_gru():
    config_space = get_gru_config_space()

    # 100 configurations
    for config_idx in range(100):
        sampled_config = dict(config_space.sample_configuration())
        sampled_config = {k: v.item() if isinstance(v, np.generic) else v for k, v in sampled_config.items()}

        for dataset_name, dataset_path in datasets.items():
            try:
                print(f"\nTraining on {dataset_name} with config {config_idx + 1}")
                
                dataset, input_shape, num_classes = load_dataset(dataset_path, dataset_name)
                print(f"Input shape: {input_shape}, Num classes: {num_classes}")

                # splitting the dataset into training and validation sets
                val_size = min(50, int(0.2 * len(dataset)))
                train_size = len(dataset) - val_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

                model = build_gru(sampled_config, input_shape, num_classes)

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

                accelerator = "mps" if torch.backends.mps.is_available() else "cpu"

                early_stopping = EarlyStopping(monitor="val_loss", patience=30, mode="min")
                trainer = pl.Trainer(accelerator=accelerator, max_epochs=1024, callbacks=[JSONLogger(metrics), EpochTimeLogger(metrics), early_stopping], log_every_n_steps=1, gradient_clip_val=1.0)

                trainer.fit(model, train_loader, val_loader)
                metrics["epochs"] = trainer.current_epoch
                save_results("GRU", dataset_name, config_idx, metrics)

            except Exception as e:
                print(f"Failed for {dataset_name} config {config_idx}: {str(e)}")
                continue

if __name__ == "__main__":
    train_gru()
