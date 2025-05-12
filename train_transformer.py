import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from config_spaces import get_transformer_config_space
from model_builder import build_transformer

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
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # normalizes the data
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # ensures that the labels are integers starting from 0
    y = y - y.min()

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
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) if X.ndim == 2 else torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    input_shape = (X_tensor.shape[1], X_tensor.shape[2])  # (seq_len, num_features)
    num_classes = len(np.unique(y))
    
    return dataset, input_shape, num_classes

# this method is used to save the results of the training
def save_results(arch_name, dataset_name, config_idx, metrics):
    result_dir = os.path.join("results", arch_name)
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, f"{dataset_name}_config_{config_idx + 1}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved: {out_path}")

# this method is used to train the transformer model
def train_transformer():
    config_space = get_transformer_config_space()

    for config_idx in range(5):
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
                val_size = max(1, int(0.2 * len(dataset)))  # ensures at least 1 sample
                train_size = len(dataset) - val_size
                
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

                batch_size = 32
                # reduces batch size for larger datasets
                if input_shape[0] > 1000:
                    batch_size = 16
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), persistent_workers=True)
                
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), persistent_workers=True)

                model = build_transformer(sampled_config, input_shape, num_classes)

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

                trainer = pl.Trainer(accelerator="cpu", max_epochs=15, callbacks=[JSONLogger(metrics)], gradient_clip_val=0.5, accumulate_grad_batches=1, log_every_n_steps=1)

                print(f"[{dataset_name}] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
                print(f"Transformer config: {sampled_config}")

                trainer.fit(model, train_loader, val_loader)
                metrics["epochs"] = trainer.current_epoch
                save_results("Transformer", dataset_name, config_idx, metrics)

            except Exception as e:
                print(f"Failed for {dataset_name} config {config_idx}: {e}")
                continue

if __name__ == "__main__":
    train_transformer()