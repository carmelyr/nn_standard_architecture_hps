import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple

# ---- Fully Connected Neural Network ---- #
class FCNN(pl.LightningModule):
    def __init__(self, config: Dict, input_size: int, output_size: int):
        super(FCNN, self).__init__()

        self.save_hyperparameters()

        # parameters from configuration space
        hidden_units = config["hidden_units"]
        num_layers = config["num_layers"]
        dropout_rate = config["dropout_rate"]
        activation_name = config.get("output_activation", "relu")
        learning_rate = config["learning_rate"]

        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}

        activation_layer = activations.get(activation_name, nn.ReLU())      # default to ReLU if not found

        # dynamically build the model
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation_layer)
            layers.append(nn.BatchNorm1d(hidden_units))

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units

       # final output layer
        layers.append(nn.Linear(hidden_units, output_size))

        # adds the final activation function
        self.model = nn.Sequential(*layers)

        # optimizer config
        self.learning_rate = learning_rate
        self.weight_decay = config.get("weight_decay", 0.0)  # default to 0.0 if not specified

        # categorical cross-entropy loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flattens the input
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    # this method is used to configure the optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

# ---- FCNN builder function ---- #
# this function builds the FCNN model based on the sampled config and dataset specs
def build_fcnn(config: Dict, input_size: int, output_size: int):
    return FCNN(config, input_size, output_size)


# ---- Convolutional Neural Network ---- #
class CNN(pl.LightningModule):
    def __init__(self, config: Dict, input_shape: Tuple[int, int, int], num_classes: int):
        super(CNN, self).__init__()

        self.save_hyperparameters()

        # parameters from configuration space
        num_filters = config["num_filters"]
        kernel_size = config["kernel_size"]
        pooling_type = config["pooling"]
        pooling_size = config["pooling_size"]
        num_layers = config.get("num_layers", 1)
        dropout_rate = config["dropout_rate"]
        activation_name = config.get("activation", "relu")
        learning_rate = config["learning_rate"]
    
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU(), "elu": nn.ELU(), "sigmoid": nn.Sigmoid()}

        activation_layer = activations.get(activation_name, nn.ReLU())  # default to ReLU if not found

        # builds the model dynamically
        layers = []
        in_channels = input_shape[-1]  # input shape is (seq_len, num_features, channels)
        seq_len = input_shape[0]        # seq_len is the first dimension

        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, stride=1, padding=kernel_size // 2))

            layers.append(activation_layer)
            layers.append(nn.BatchNorm1d(num_filters))
            
            # pooling layer is added only if seq_len is large enough
            if seq_len > 4:
                if pooling_type == "max":
                    layers.append(nn.MaxPool1d(pooling_size))
                else:
                    layers.append(nn.AvgPool1d(pooling_size))
                seq_len = seq_len // 2
                
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            in_channels = num_filters

        self.cnn_layers = nn.Sequential(*layers)

        # this block calculates the size of the flattened output after the CNN layers
        # it is used to define the input size of the final classifier
        # the input shape is (batch_size, channels, seq_len)
        with torch.no_grad():
            sample_input = torch.zeros(2, input_shape[-1], input_shape[0])
            self.cnn_layers.eval()
            output = self.cnn_layers(sample_input)
            flatten_size = output.view(2, -1).shape[1]
            self.cnn_layers.train()

        # final classifier
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flatten_size, num_classes))

        # optimizer paraemters
        self.learning_rate = learning_rate
        self.weight_decay = config.get("weight_decay", 0.0)

        # cross-entropy loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.ndim == 3:
            x = x.permute(0, 2, 1)      # (batch, channels, seq_len)
        return self.classifier(self.cnn_layers(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # this ensures that metrics are always logged
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # ensures that metrics are always logged for validation
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return {'val_loss': loss, 'val_accuracy': acc}

    # this method is used to configure the optimizer, called by the trainer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

# ---- CNN builder function ---- #
def build_cnn(config: Dict, input_shape: Tuple[int, int, int], num_classes: int):
    return CNN(config, input_shape, num_classes)


# ---- Long Short-Term Memory ---- #
class LSTM(pl.LightningModule):
    def __init__(self, config: Dict, input_shape: Tuple[int, int], num_classes: int):
        super(LSTM, self).__init__()

        self.save_hyperparameters()

        # parameters from configuration space
        hidden_units = config["hidden_units"]
        num_layers = config["num_layers"]
        dropout_rate = config["dropout_rate"]
        learning_rate = config["learning_rate"]
        bidirectional = config.get("bidirectional", False)
        activation_name = config.get("activation", "tanh")
        weight_decay = config.get("weight_decay", 0.0)

        # checks if bidirectional is a numpy boolean and converts it to a native Python boolean
        if isinstance(bidirectional, np.bool_):
            bidirectional = bool(bidirectional)

        # LSTM input shape: (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=hidden_units, num_layers=num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0, bidirectional=bidirectional)

        directions = 2 if bidirectional else 1

        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}

        activation_layer = activations.get(config.get("output_activation", "tanh"), nn.Tanh())  # default to Tanh if not found

        # classifier of the LSTM
        self.classifier = nn.Sequential(nn.Linear(hidden_units * directions, hidden_units), activation_layer, nn.Dropout(dropout_rate), nn.Linear(hidden_units, num_classes))

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        x = x.float()
        
        # passes input through LSTM
        # LSTM output shape: (batch, seq_len, hidden_size * directions)
        # hidden state shape: (num_layers * directions, batch, hidden_size)
        lstm_out, (hn, _) = self.lstm(x)
        
        # if bidirectional, it concatenates the last hidden states from both directions
        # otherwise, it takes the last hidden state from the last layer
        if self.lstm.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hn = hn[-1]
            
        return self.classifier(hn)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # checks for NaN values in logits
        if torch.isnan(logits).any():
            print("NaN detected in logits!")
            return None
        
        loss = self.loss_fn(logits, y)
        
        # checks for NaN values in loss
        if torch.isnan(loss):
            print("NaN detected in loss!")
            return None
        
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

def build_lstm(config: Dict, input_shape: Tuple[int, int], num_classes: int):
    return LSTM(config, input_shape, num_classes)


# ---- Gated Recurrent Unit ---- #
class GRU(pl.LightningModule):
    def __init__(self, config: dict, input_shape: tuple, output_size: int):
        super().__init__()
        self.save_hyperparameters()

        # input shape is (batch, seq_len, features)
        seq_len, num_features = input_shape if len(input_shape) == 2 else (input_shape[0], 1)
        
        # parameters from configuration space
        hidden_size = config["hidden_units"]
        num_layers = config["num_layers"]
        dropout = config["dropout_rate"]
        learning_rate = config["learning_rate"]
        weight_decay = config.get("weight_decay", 0.0)
        output_activation_name = config.get("output_activation", "relu")

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "sigmoid": nn.Sigmoid(),
            "linear": nn.Identity()
        }
        output_activation = activations.get(output_activation_name, nn.ReLU())

        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=config.get("bidirectional", False))

        # calculates the size of the classifier input
        # if bidirectional, it doubles the hidden size
        classifier_input_size = hidden_size * (2 if config.get("bidirectional", False) else 1)
        
        # classifier of the GRU
        self.classifier = nn.Sequential(nn.Linear(classifier_input_size, hidden_size), output_activation, nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        x = x.float()
        
        # checks if the input is 2D and adds a feature dimension if necessary
        if x.ndim == 2:
            x = x.unsqueeze(-1)
            
        gru_out, _ = self.gru(x)  # gru_out shape: (batch, seq_len, hidden_size)
        
        # takes the last output of the GRU
        # last_output shape: (batch, hidden_size) if bidirectional, else (batch, hidden_size * num_layers)
        last_output = gru_out[:, -1, :]
        
        return self.classifier(last_output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

# ---- GRU builder function ---- #
def build_gru(config: dict, input_shape: tuple, output_size: int):
    return GRU(config, input_shape, output_size)


# ---- Transformer ---- #
class TransformerModel(pl.LightningModule):
    def __init__(self, config: Dict, input_shape: Tuple[int, int], num_classes: int):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()

        # parameters from configuration space
        num_heads = config["num_heads"]
        hidden_units = config["hidden_units"]
        ff_dim = config["ff_dim"]
        num_layers = config["num_layers"]
        pooling = config.get("pooling", "mean")
        dropout_rate = config["dropout_rate"]
        learning_rate = config["learning_rate"]
        activation_name = config.get("activation", "relu")
        weight_decay = config.get("weight_decay", 0.0)

        # ensures that hidden_units is divisible by num_heads by adjusting its value if needed
        if hidden_units % num_heads != 0:
            hidden_units = ((hidden_units // num_heads) + 1) * num_heads
            print(f"Adjusted hidden_units to {hidden_units} to be divisible by num_heads")

        activation_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}[activation_name]

        # Transformer encoder layer
        class TransformerEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward, dropout, activation_fn):
                super().__init__()

                # Multihead-Attention layer
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(dim_feedforward, d_model)

                # Layer Normalization and Dropout layers
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)

                self.activation = activation_fn

            # src2 is Multihead-Attention output
            # src is the input to the layer
            # src_mask is the attention mask
            # is_causal is a boolean indicating if the attention should be causal
            # src_key_padding_mask is the key padding mask that is used to ignore certain positions in the input
            def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
                src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal)[0]
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)
                return src

        # Transformer encoder layer
        encoder_layer = TransformerEncoderLayer(d_model=hidden_units, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate, activation_fn=activation_fn)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Linear(input_shape[1], hidden_units)
        self.pooling = pooling
        self.classifier = nn.Linear(hidden_units, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)      # (seq_len, batch, hidden)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)      # (batch, seq_len, hidden)

        # applies pooling
        # if pooling is "mean", it averages the outputs across the sequence length
        # otherwise, it takes the maximum output across the sequence length
        if self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            x = x.max(dim=1).values
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

# ---- Transformer builder function ---- #
def build_transformer(config: Dict, input_shape: Tuple[int, int], num_classes: int):
    return TransformerModel(config, input_shape, num_classes)
