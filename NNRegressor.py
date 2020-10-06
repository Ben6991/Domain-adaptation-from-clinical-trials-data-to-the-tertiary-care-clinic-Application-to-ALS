import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models.EarlyStopping import EarlyStopping


class NNRegressorModel:
    model: nn.Module

    def __init__(self, input_size, hidden_sizes=[1], activation='relu', dropout=0.0, batch_norm=False, use_cuda=False):

        self.use_cuda = use_cuda
        self.model = NNRegressor(input_size, hidden_sizes, activation, True, batch_norm, dropout)
        self.dev = torch.device('cpu')
        self.batch_norm = batch_norm
        if self.use_cuda and torch.cuda.is_available():
            self.dev = torch.device('cuda')
            self.model.to(self.dev)
            print("using cuda")
        self.optimizer = None
        self.criterion = None

    def compile(self, optimizer='adam', learning_rate=0.001, l1=0.0):
        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=l1)
        if optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=learning_rate, weight_decay=l1)
        self.criterion = nn.MSELoss()

    def fit(self, x_train, y_train, validation_data=None, epochs=1, batch_size=32, model_path=None):

        early_stopping = EarlyStopping(model_path, patience=20, verbose=False)

        X_val, y_val = None, None
        if validation_data is not None:
            X_val, y_val = validation_data

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.FloatTensor(x_train).to(self.dev)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.dev)

        if not isinstance(X_val, torch.Tensor):
            X_val = torch.FloatTensor(X_val).to(self.dev)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(self.dev)

        train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)

        for epoch in range(1, epochs + 1):
            s = time.time()
            self.model.train()
            for x_batch, y_batch in train_loader:
                if not (x_batch.shape[0] == 1 and self.batch_norm):
                    self.optimizer.zero_grad()
                    y_preds = self.model(x_batch)
                    loss = self.criterion(y_preds, y_batch)
                    loss.backward()
                    self.optimizer.step()
            with torch.no_grad():
                # train loss
                self.model.eval()
                train_preds = self.model(x_train)
                train_loss = self.criterion(train_preds, y_train)

                # validation loss
                val_preds = self.model(X_val)
                val_loss = self.criterion(val_preds, y_val)

            epoch_time = time.time() - s
            print(f"Epoch {epoch} train: {train_loss.item():.2f} val: {val_loss.item():.2f} secs: {epoch_time:.3f}",
                  flush=True)

            # Early stopping
            early_stopping(val_loss.item(), self.model)
            if early_stopping.early_stop:
                print(f"Early stop!! best epoch: {early_stopping.best_epoch} with {early_stopping.best_score * -1:.2f}",
                      flush=True)
                self.model = early_stopping.load_best_model(self.model)
                break

    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.dev)
        with torch.no_grad():
            pred = self.model(x)
        return pred.detach().cpu().numpy()


class NNRegressor(nn.Module):
    layers: Union[Callable[[Tuple[Any, ...], Dict[str, Any]], Any], list]

    def __init__(self, input_size: int, hidden_sizes=None, activation=None, bias=True, batch_norm=True, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(hidden_sizes)):
            dense = nn.Linear(input_size if i == 0 else hidden_sizes[i - 1],
                              hidden_sizes[i], bias)
            self.layers.append(dense)

            if activation == 'tanh':
                self.layers.append(nn.Tanh())
                self.layers.append(nn.Dropout(dropout))
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
                self.layers.append(nn.Dropout(dropout))
            else:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))

        # ------------------------------------------------
        #       BatchNorm
        # ------------------------------------------------
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(num_features=hidden_sizes[-1], affine=True))

        # ------------------------------------------------
        #       Output layer
        # ------------------------------------------------
        self.layers.append(nn.Linear(hidden_sizes[-1], 1, bias))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
