import os
import json
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from config_spaces import get_lstm_config_space, lstm_seed
from model_builder import build_lstm
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

    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

    elif path.endswith(".txt"):
        data = np.loadtxt(path)
        y = data[:, 0].astype(int)
        X = data[:, 1:]

    else:
        raise ValueError(f"Unsupported file format: {path}")

    # handles NaN and infinite values
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    X = np.where(np.isposinf(X), 1e6, X)
    X = np.where(np.isneginf(X), -1e6, X)
    
    # normalizes the data
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # ensures labels start from 0
    y = y - y.min()

    # data augmentation for small datasets
    if X.shape[0] < 100:
        def jitter(x, sigma=0.03):
            return x + np.random.normal(0, sigma, size=x.shape)
        
        X_aug = np.array([jitter(x) for x in X])
        y_aug = y.copy()
        X = np.concatenate([X, X_aug])
        y = np.concatenate([y, y_aug])

    # converts the data to PyTorch tensors and handles the feature dimension
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) if X.ndim == 2 else torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    input_shape = (X_tensor.shape[1], X_tensor.shape[2])  # (seq_len, num_features)
    num_classes = len(np.unique(y))
    
    return dataset, input_shape, num_classes

# this method saves the results to a JSON file
def save_results(arch_name, dataset_name, config_idx, metrics, lstm_seed):
    # directory structure
    config_id = config_idx + 1
    result_dir = os.path.join("results", arch_name, dataset_name, f"config_{config_id}")
    os.makedirs(result_dir, exist_ok=True)

    # full filename
    filename = f"{dataset_name}_config_{config_id}_seed{lstm_seed}.json"
    out_path = os.path.join(result_dir, filename)

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✔ Saved: {out_path}")

# this method trains the LSTM model
def train_lstm():
    config_space = get_lstm_config_space()

    # 100 configurations
    for config_idx in range(100):
        sampled_config = dict(config_space.sample_configuration())
        sampled_config = {k: v.item() if isinstance(v, np.generic) else v for k, v in sampled_config.items()}

        for dataset_name, dataset_path in datasets.items():
            try:
                dataset, input_shape, num_classes = load_dataset(dataset_path, dataset_name)
                
                # skips very small datasets to ensure that the dataset is large enough for training
                if len(dataset) < 10:
                    print(f"Skipping {dataset_name} - too small ({len(dataset)} samples)")
                    continue
                
                # splitting the dataset into training and validation sets
                val_size = max(1, int(0.2 * len(dataset)))  # ensures at least 1 sample is in the validation set
                train_size = len(dataset) - val_size
                
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=min(4, os.cpu_count()), persistent_workers=True)
                
                val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=min(4, os.cpu_count()), persistent_workers=True)

                model = build_lstm(sampled_config, input_shape, num_classes)

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
                trainer = pl.Trainer(accelerator="cpu", max_epochs=1024, callbacks=[JSONLogger(metrics), EpochTimeLogger(metrics), early_stopping], gradient_clip_val=1.0, accumulate_grad_batches=2, log_every_n_steps=1)

                print(f"[{dataset_name}] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

                trainer.fit(model, train_loader, val_loader)
                metrics["epochs"] = trainer.current_epoch
                save_results("LSTM", dataset_name, config_idx, metrics, lstm_seed)

            except Exception as e:
                print(f"Failed for {dataset_name} config {config_idx}: {e}")
                continue

if __name__ == "__main__":
    train_lstm()