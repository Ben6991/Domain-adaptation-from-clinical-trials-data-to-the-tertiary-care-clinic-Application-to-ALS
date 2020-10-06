from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models.EarlyStopping import EarlyStopping


class RNNRegressorModel:
    """
    RNN-based model for multivariate unevenly spaced time series tasks
    """
    ""

    def __init__(self, input_size,
                 rnn_hidden_units,
                 rnn_bias=True,
                 ff_bias=True,
                 ff_hidden=None,
                 rnn_type='lstm',
                 learn_initial_state=False,
                 activation='tanh',
                 use_cuda=False,
                 num_workers=0,
                 norm_type='batch'):

        self.num_workers = num_workers
        self.use_cuda = use_cuda
        self.optimizer = None
        self.criterion = None
        self.train_losses = []
        self.val_losses = []
        self.norm_type = norm_type
        self.early_stopping_obj = None
        self.model = RNNRegressor(input_size=input_size,
                                  rnn_hidden_units=rnn_hidden_units,
                                  rnn_bias=rnn_bias,
                                  ff_bias=ff_bias,
                                  ff_hidden=ff_hidden,
                                  rnn_type=rnn_type,
                                  learn_initial_state=learn_initial_state,
                                  activation=activation,
                                  use_cuda=use_cuda,
                                  norm_type=norm_type)

    def compile(self, optimizer='adam', lr=0.001, loss='mse', weight_decay=0.0):
        if loss == 'mse':
            self.criterion = nn.MSELoss()
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, X_train, y_train, lengths,
            validation_data=None, batch_size=32,
            epochs=1, early_stopping=None,
            padding_value=-999, verbose=1,
            use_cuda=False):

        dev = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        if use_cuda:
            self.model = self.model.to(dev)
        if validation_data is not None:
            X_val, y_val, val_lengths = validation_data
            X_val_padded = pad_sequence(X_val, batch_first=True, padding_value=-999)
            X_val_packed = pack_padded_sequence(X_val_padded.to(dev), val_lengths.to(dev), True, False)
            if isinstance(y_val, list):
                y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(dev)

        if early_stopping is not None and self.early_stopping_obj is None:
            early_stopping_obj = EarlyStopping(path=early_stopping['path'],
                                               patience=early_stopping['patience'],
                                               verbose=early_stopping['verbose'],
                                               delta=early_stopping['delta'])
            self.early_stopping_obj = early_stopping_obj

        X_train_padded = pad_sequence(X_train, batch_first=True, padding_value=padding_value)

        # if y_train is list of tensors convert it to tensor size (N, 1)
        if isinstance(y_train, list):
            y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(dev)
        dff = TensorDataset(X_train_padded, y_train, lengths)
        data_loader = DataLoader(dff, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        X_train_packed = pack_padded_sequence(X_train_padded.to(dev), lengths.to(dev), True, False)

        for epoch in range(epochs):
            start_time = time()
            self.model.train()
            for x_batch, y_batch, len_batch in data_loader:
                if not (x_batch.shape[0] == 1 and self.norm_type == 'batch'):
                    self.optimizer.zero_grad()
                    x_batch = x_batch.to(dev)
                    x_packed = pack_padded_sequence(x_batch, len_batch.to(dev), batch_first=True, enforce_sorted=False)
                    preds = self.model(x_packed)
                    loss = self.criterion(preds, y_batch.to(dev))
                    loss.backward()
                    self.optimizer.step()
            end_time = time()
            with torch.no_grad():
                self.model.eval()
                preds = self.model(X_train_packed)
                train_loss = self.criterion(preds, y_train)
                self.train_losses.append(train_loss.item())
                if validation_data is not None:
                    preds = self.model(X_val_packed)
                    val_loss = self.criterion(preds, y_val)
                    self.val_losses.append(val_loss.item())

                if verbose == 1:
                    print(
                        f"Epoch {epoch + 1}/{epochs} Train: {train_loss.item():.3f} Validation: {val_loss.item():.3f} "
                        f"time: {(end_time - start_time):.2f} sec", flush=True)
                if self.early_stopping_obj is not None:
                    self.early_stopping_obj(val_loss, self.model)
                    if self.early_stopping_obj.early_stop:
                        print(f'Early stopping!!! best epoch: {self.early_stopping_obj.best_epoch}', flush=True)
                        self.model = self.early_stopping_obj.load_best_model(self.model)
                        break

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

    def predict(self, X, lengths, padding_value=-999, numpy=True):
        dev = torch.device("cuda" if torch.cuda.is_available() and self.model.use_cuda else "cpu")
        X_padded = pad_sequence(X, batch_first=True, padding_value=padding_value).to(dev)
        X_packed = pack_padded_sequence(X_padded.to(dev), lengths.to(dev), True, False)
        X_packed = X_packed.to(dev)
        return self.model(X_packed.to(dev)).detach().cpu().numpy() if numpy else self.model(X_packed)


##############################################
#           Pytorch module                   #
##############################################

def initialize_weights(layer):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)


class RNNRegressor(nn.Module):
    def __init__(self,
                 input_size,
                 rnn_hidden_units,
                 rnn_bias=True,
                 ff_bias=True,
                 ff_hidden=None,
                 rnn_type='lstm',
                 learn_initial_state=False,
                 activation='tanh',
                 use_cuda=False,
                 norm_type='batch'):  # TODO
        super().__init__()
        self.use_cuda = use_cuda
        self.first_rnn_units = rnn_hidden_units[0]
        self.learn_initial_state = learn_initial_state
        self.rnn_type = rnn_type
        # Recurrent block
        rnn_layers = nn.ModuleList()
        for i in range(len(rnn_hidden_units)):

            # LSTM
            if rnn_type == 'lstm':
                rnn_layer = nn.LSTM(input_size=int(input_size) if i == 0 else int(rnn_hidden_units[i - 1]),
                                    hidden_size=int(rnn_hidden_units[i]),
                                    num_layers=1,
                                    bias=rnn_bias,
                                    batch_first=True)

                initialize_weights(rnn_layer)

            # RNN
            elif rnn_type == 'rnn':
                rnn_layer = nn.RNN(input_size=input_size if i == 0 else rnn_hidden_units[i - 1],
                                   hidden_size=rnn_hidden_units[i],
                                   num_layers=1,
                                   bias=rnn_bias,
                                   batch_first=True,
                                   nonlinearity=activation)

                initialize_weights(rnn_layer)

            # GRU
            elif rnn_type == 'gru':
                rnn_layer = nn.GRU(input_size=input_size if i == 0 else rnn_hidden_units[i - 1],
                                   hidden_size=rnn_hidden_units[i],
                                   num_layers=1,
                                   bias=rnn_bias,
                                   batch_first=True)
                initialize_weights(rnn_layer)

            rnn_layers.append(rnn_layer)

        self.rnn_layers = rnn_layers

        ff_layers = None
        # Feed forward block
        if len(ff_hidden) > 0:
            ff_layers = nn.ModuleList()
            for i in range(len(ff_hidden)):

                # BATCH NORM
                if norm_type == 'batch':
                    batch_norm = nn.BatchNorm1d(num_features=rnn_hidden_units[-1])
                    ff_layers.append(batch_norm)

                ff_layer = nn.Linear(in_features=rnn_hidden_units[-1] if i == 0 else ff_hidden[i - 1],
                                     out_features=ff_hidden[i],
                                     bias=ff_bias)
                ff_layers.append(ff_layer)

        self.ff_layers = ff_layers

        self.h_0_layer = None
        self.c_0_layer = None
        if learn_initial_state:
            self.h_0_layer = nn.Linear(rnn_hidden_units[0], rnn_hidden_units[0], bias=False)
            if rnn_type == 'lstm':
                self.c_0_layer = nn.Linear(rnn_hidden_units[0], rnn_hidden_units[0], bias=False)

    def forward(self, x):
        # get batch size
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            batch_size = x.batch_sizes[0].item()
        else:
            batch_size = x.shape[0]

        h_0 = torch.zeros(1, batch_size, self.first_rnn_units)
        c_0 = torch.zeros(1, batch_size, self.first_rnn_units)

        if self.use_cuda:
            h_0 = h_0.to(torch.device('cuda'))
            c_0 = c_0.to(torch.device('cuda'))

        if self.learn_initial_state:
            h_0 += 1
            h_0 = self.h_0_layer(h_0)
            if self.rnn_type == 'lstm':
                c_0 += 1
                c_0 = self.c_0_layer(c_0)

        h_n = None
        # RNN layers
        for rnn_layer in self.rnn_layers:
            if self.rnn_type == 'lstm':
                x, (h_n, c_n) = rnn_layer(x, (h_0, c_0))
            elif self.rnn_type == 'rnn' or self.rnn_type == 'gru':
                x, h_n = rnn_layer(x, h_0)
        h_n.squeeze_(0)
        # FF layers
        for ff_layer in self.ff_layers:
            h_n = ff_layer(h_n)
        return h_n
