import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple
import math

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
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)

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
        self.num_layers = config["num_layers"]
        dropout_rate = config["dropout_rate"]
        self.learning_rate = config["learning_rate"]
        bidirectional = config.get("bidirectional", False)
        
        # main LSTM layer
        # input shape to the model is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_shape[1],      # number of features per timestep
            hidden_size=hidden_units,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate if self.num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        directions = 2 if bidirectional else 1

        # classifier head
        self.classifier = nn.Sequential(nn.Linear(hidden_units * directions, hidden_units * 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_units * 2, num_classes))
        
        # applies weight initialization
        self.apply(self._init_weights)
        
        # cross-entropy loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    # this method initializes the weights of the model
    def _init_weights(self, module):

        # sets weights with a uniform distribution that keeps the scale of gradients roughly the same in all layers
        # biases are initialized to zero
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():

                # weights that connect the input vector to the LSTM gates
                # between input and hidden layers
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)

                # recurrent weights, connecting the hidden state to itself over time
                # helps to preserve the gradient over long sequences
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)

                # LSTM gates are ored in order: [input_gate | forget_gate | cell_gate | output_gate]
                # each bias vector is split into 4 parts, where each corresponds to a gate
                # param.size(0) gives the total size
                # n//4 : n//2 is the forget gate’s slice
                # forget gate bias is set to 1 to encourage the LSTM to remember things in early training
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[(n // 4):(n // 2)].fill_(1)

    def forward(self, x):
        x = x.float()
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # takes last timestep
        return self.classifier(last_output)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': acc}

    # this method is used to configure the optimizer, called by the trainer
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.config.get("weight_decay", 0.0))

        # learning rate scheduler
        # reduces the learning rate when a metric has stopped improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

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

        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU(), "elu": nn.ELU(), "sigmoid": nn.Sigmoid(), "linear": nn.Identity()}

        output_activation = activations.get(output_activation_name, nn.ReLU())

        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=config.get("bidirectional", False))

        # calculates the size of the classifier input
        # if bidirectional, it doubles the hidden size
        classifier_input_size = hidden_size * (2 if config.get("bidirectional", False) else 1)
        
        # classifier of the GRU
        self.classifier = nn.Sequential(nn.LayerNorm(classifier_input_size), nn.Linear(classifier_input_size, hidden_size), output_activation, nn.Dropout(dropout), nn.Linear(hidden_size, output_size))

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # sets forget gate bias to 1
                n = param.size(0)
                param.data[(n // 3):(2 * n // 3)].fill_(1)  # GRU has 3 gates
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
# ---- GRU builder function ---- #
def build_gru(config: dict, input_shape: tuple, output_size: int):
    return GRU(config, input_shape, output_size)


# ---- Transformer ---- #
class TransformerModel(pl.LightningModule):
    # input format is (sequence_length, num_features)
    # input to the model will be shaped like (batch_size, sequence_length, num_features)
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
        activation_name = config.get("activation", "gelu")
        weight_decay = config.get("weight_decay", 0.0)

        # ensures that hidden_units is divisible by num_heads for multi-head attention
        self.hidden_units = ((hidden_units // num_heads) + 1) * num_heads
        
        self.activation = nn.GELU()
        
        # positional encoding encodes the position of each element in the sequence
        # added to the input embeddings to provide information about the order of the sequence
        # generates positional vectors (sine/cosine) and adds them to the input
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 5000):
                super().__init__()
                self.d_model = d_model                          # dimension of the model

                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(1, max_len, d_model)
                pe[0, :, 0::2] = torch.sin(position * div_term)

                if d_model % 2 == 0:
                    pe[0, :, 1::2] = torch.cos(position * div_term)         # for even d_model
                else:
                    pe[0, :, 1::2] = torch.cos(position * div_term[:-1])    # for odd d_model
                self.register_buffer('pe', pe)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # ensures that input dimensions match
                if x.size(2) != self.d_model:
                    raise ValueError(f"Input feature dimension {x.size(2)} doesn't match positional encoding dimension {self.d_model}")
                return x + self.pe[:, :x.size(1)]

        # embedding layer that projects the input features to the hidden units
        self.embedding = nn.Sequential(
            nn.Linear(input_shape[1], self.hidden_units),
            nn.LayerNorm(self.hidden_units),
            nn.Dropout(dropout_rate),
            PositionalEncoding(self.hidden_units, input_shape[0])
        )

        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_units,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(self.hidden_units),
            nn.Linear(self.hidden_units, num_classes)
        )

        # weights initialization
        self.apply(self._init_weights)
        
        # optimizer parameters and loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pooling = pooling

    # this method initializes the weights of the model
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
# ---- Transformer builder function ---- #
def build_transformer(config: Dict, input_shape: Tuple[int, int], num_classes: int):
    return TransformerModel(config, input_shape, num_classes)