import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from ModelComparison.Models.EarlyStopping import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from time import time
import matplotlib.pyplot as plt


class PaddingLayer(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (0, self.padding))


class DCN(nn.Module):
    def __init__(self, n_features, n_filters, kernel_size, dilation, static_features_to_add=0, n_hidden_dense=[]):
        """
        :param n_features: int
        number of temporal features (out channels of the first conv layer)s
        :param n_filters:   list of integers (temporal depth length)
        n_filters[i] is number of filters of the i-th conv layer
        :param kernel_size: int
        kernel size in each conv layer
        :param n_hidden_dense: list of integers
        n_hidden_dense[i] is number of hidden units in the i-th layer
        :param dilation:
        dilation[i] is the dilation of the DCN's i-th conv layer
        """
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i in range(len(n_filters)):
            # Add padding layer to make sure that the output shape match the input shape
            self.conv_layers.append(PaddingLayer((kernel_size-1)*dilation[i]))
            self.conv_layers.append(nn.Conv1d(in_channels=n_filters[i-1] if i != 0 else n_features - static_features_to_add,
                                              out_channels=n_filters[i],
                                              kernel_size=kernel_size,
                                              dilation=dilation[i],
                                              bias=False))
            self.conv_layers.append(nn.ReLU())

        for i in range(len(n_hidden_dense)):
            self.fc_layers.append(nn.Linear(in_features=n_filters[-1] + static_features_to_add if i == 0 else n_hidden_dense[i-1],
                                            out_features=n_hidden_dense[i]))
            self.fc_layers.append(nn.ReLU()) if i < len(n_hidden_dense) -1 else None

        self.temporal_block = nn.Sequential(*self.conv_layers)
        self.fully_connected_block = nn.Sequential(*self.fc_layers)

        # self.init_weights()
        self.receptive_field = max(dilation) * kernel_size
        self.early_stopping_obj = None
        self.static_features = static_features_to_add

    def init_weights(self):
        for layer in self.temporal_block:
            if type(layer) == nn.Conv1d:
                layer.weight.data.normal_(0, 0.01)
        for layer in self.fully_connected_block:
            if type(layer) == nn.Linear:
                layer.weight.data.normal_(0, 0.01)

    def forward(self, x, x_static=None, encode_x=False):
        x = self.temporal_block(x)
        x = x[:, :, 0].contiguous()
        if encode_x:
            return x
        if x_static is not None:
            x = torch.cat((x, x_static), 1)
        x = self.fully_connected_block(x)
        return x


    def fit(self,
            X_train, y_train,
            validation_data=None,
            batch_size=32,
            epochs=1,
            early_stopping=None,
            verbose=1,
            use_cuda=False):

        dev = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        if use_cuda:
            self.to(dev)
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_padded = pad_sequence(X_val, batch_first=True).transpose(1, 2).to(dev)
            if self.static_features > 0:
                x_val_static = X_val_padded[:, -self.static_features:, 0].to(dev)
                X_val_padded = X_val_padded[:, :-self.static_features, :].to(dev)
            else:
                x_val_static = None

            if isinstance(y_val, list):
                y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(dev)

        if early_stopping is not None and self.early_stopping_obj is None:
            early_stopping_obj = EarlyStopping(path=early_stopping['path'],
                                               patience=early_stopping['patience'],
                                               verbose=early_stopping['verbose'],
                                               delta=early_stopping['delta'])
            self.early_stopping_obj = early_stopping_obj

        # if y_train is list of tensors convert it to tensor size (N, 1)
        if isinstance(y_train, list):
            y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(dev)

        X_train_padded = pad_sequence(X_train, batch_first=True).transpose(1, 2).to(dev)
        dff = TensorDataset(X_train_padded, y_train)
        data_loader = DataLoader(dff, batch_size=batch_size, shuffle=True)
        if self.static_features > 0:
            x_train_static = X_train_padded[:, -self.static_features:, 0].to(dev)
            X_train_padded = X_train_padded[:, :-self.static_features, :].to(dev)
        else:
            x_train_static = None

        for epoch in range(epochs):
            start_time = time()
            for x_batch, y_batch in data_loader:
                self.__optimizer.zero_grad()
                x_batch_padded = pad_sequence(x_batch.to(dev), batch_first=True).to(dev)
                if self.static_features > 0:
                    static_batch = x_batch_padded[:, -self.static_features:, 0].to(dev)
                    x_batch_padded = x_batch_padded[:, :-self.static_features, :].to(dev)
                else:
                    static_batch=None
                preds = self(x_batch_padded, x_static=static_batch)
                loss = self.__criterion(preds, y_batch.to(dev))
                loss.backward()
                self.__optimizer.step()
            end_time = time()
            with torch.no_grad():
                preds = self(X_train_padded, x_static=x_train_static)
                train_loss = self.__criterion(preds, y_train)
                self.train_losses.append(train_loss.item())
                if validation_data is not None:
                    preds = self(X_val_padded, x_static=x_val_static)
                    val_loss = self.__criterion(preds, y_val)
                    self.val_losses.append(val_loss.item())
                if verbose == 1:
                    if validation_data is None:
                        to_print = f"Epoch {epoch + 1}/{epochs} Train: {train_loss.item():.3f} time: {(end_time-start_time):.2f} sec"
                    else:
                        to_print = f"Epoch {epoch + 1}/{epochs} Train: {train_loss.item():.3f} Validation: {val_loss.item():.3f} time: {(end_time-start_time):.2f} sec"
                    print(to_print, flush=True)
                if self.early_stopping_obj is not None:
                    self.early_stopping_obj(val_loss, self)
                    if self.early_stopping_obj.early_stop:
                        print(f'Early stopping!!! best epoch: {self.early_stopping_obj.best_epoch}', flush=True)
                        break


    def compile(self, optimizer='adam', lr=0.001, loss='mse'):
        if loss == 'mse':
            self.__criterion = nn.MSELoss()
        if optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if optimizer == 'SGD':
            self.__optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def predict(self, X, numpy=True):
        self.to(torch.device('cpu'))
        X_padded = pad_sequence(X, batch_first=True).transpose(1, 2)
        return self(X_padded).detach().numpy() if numpy else self(X_padded)


    def plot_history(self):
        if len(self.train_losses) > 0:
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses)
            if len(self.val_losses) > 0:
                plt.plot(range(1, len(self.val_losses) + 1), self.val_losses)
                plt.legend(['Train', 'Validation'])
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.show()
        else:
            print(f'No training has occurred yet')
