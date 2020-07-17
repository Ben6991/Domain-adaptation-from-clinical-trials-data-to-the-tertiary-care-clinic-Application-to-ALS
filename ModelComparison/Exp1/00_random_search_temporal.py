# os.chdir('')
# sys.path.append('')
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler

sys.path.append('/home/boaz/ben')
from models.RNNRegressor import RNNRegressorModel
from models.TransformerRegressor import TransformerRegressorModel
import torch


# torch.set_num_threads(1)


def transformer_param_dist():
    param_dist = dict(lr=[0.001, 0.01, 0.1],
                      batch_size=2 ** np.arange(8, 10),
                      num_layers=[1, 2],
                      nhead=[1, 2],  # add 2
                      dim_feedforward=[16, 32, 64, 128, 256, 512, 1024])
    return param_dist


def get_param_dist(model_name):
    param_dist = None
    if model_name == 'rnn' or model_name == 'gru' or model_name == 'lstm':
        param_dist = rnn_param_dist(model_name)

    if model_name == 'transformer':
        param_dist = transformer_param_dist()
    return param_dist


def rnn_param_dist(rnn_type):
    param_dist = dict()
    if rnn_type == 'rnn':
        param_dist = dict(rnn_type=[rnn_type],
                          lr=[0.0001, 0.001, 0.01],
                          hidden_size=[8, 16, 32, 64, 128, 256, 512, 800],
                          batch_size=2 ** np.arange(4, 9),
                          num_layers=[1, 2],
                          learn_initial_state=[True, False],
                          # activation=['tanh'],
                          weight_decay=[0.001, 0.01, 0.1, 0.0, 1.],
                          norm_type=[None, 'batch'])
    if rnn_type == 'lstm':
        param_dist = dict(rnn_type=[rnn_type],
                          lr=[0.001, 0.01, 0.03],
                          hidden_size=2 ** np.arange(3, 11, 1),
                          batch_size=2 ** np.arange(7, 12),
                          num_layers=[1, 2],
                          learn_initial_state=[True, False],
                          weight_decay=[0.001, 0.01, 0.1, 0.0, 1.],
                          norm_type=[None, 'batch'])

    if rnn_type == 'gru':
        param_dist = dict(rnn_type=[rnn_type],
                          lr=[0.001, 0.01, 0.03],
                          hidden_size=[8, 16, 32, 64, 128, 256, 512],
                          batch_size=2 ** np.arange(5, 9),
                          num_layers=[1, 2, 3],
                          learn_initial_state=[True, False],
                          weight_decay=[0.001, 0.01, 0.1, 1.],
                          norm_type=[None, 'batch'])
    return param_dist


def get_sequences_lengths(X):
    return torch.FloatTensor([x.shape[0] for x in X]).reshape(-1)


def get_data():
    """
    return train, early stopping and validation tuples (X, y, lengths)
    """
    X_train, y_train, len_train = None, None, None
    X_early, y_early, len_early = None, None, None
    X_val, y_val, len_val = None, None, None
    # X_test, y_test, len_test = None, None, None

    data_path, splits_info_path = '', ''
    if data_name == 'tasmc':
        data_path = 'data/tasmc_temporal/tasmc_temporal.pkl'
        splits_info_path = '../f11_tasmc_split_info.pkl'
    if data_name == 'proact':
        data_path = 'data/proact_temporal/proact_temporal.pkl'
        splits_info_path = '../f01_proact_split_info.pkl'

    X, y, l, idx_map = pickle.load(open(data_path, 'rb'))
    splits_info = pickle.load(open(splits_info_path, 'rb'))

    train_pats = splits_info[permutation_number]['train_pats']
    early_pats = splits_info[permutation_number]['early_pats']
    val_pats = splits_info[permutation_number]['val_pats']
    # test_pats = splits_info[number_iter]['test_pats']

    print("get X train...", end=" ")
    X_train, y_train = get_partial_x(X, y, idx_map, train_pats, np.inf)
    print("ok!")
    print("get X early stopping...", end=" ")
    X_early, y_early = get_partial_x(X, y, idx_map, early_pats, 4)
    print("ok!")
    print("get X validation...", end=" ")
    X_val, y_val = get_partial_x(X, y, idx_map, val_pats, 4)
    print("ok!")
    # X_test = get_partial_x(X, y, idx_map, val_pats, np.inf)

    len_train = get_sequences_lengths(X_train)
    len_early = get_sequences_lengths(X_early)
    len_val = get_sequences_lengths(X_val)

    train = (X_train, y_train, len_train)
    early = (X_early, y_early, len_early)
    val = (X_val, y_val, len_val)
    # test = (X_test, y_test, len_test)

    return train, early, val


def get_partial_x(X, y, idx_map, pats, max_observations, random_state=123):
    random.seed(random_state)
    indices = []
    for ID in pats:
        indices += random.sample(idx_map[ID], k=min(len(idx_map[ID]), max_observations))
    indices.sort()
    return [X[i] for i in indices], [y[i] for i in indices]


def print_config(config):
    to_print = ''
    for hp in [*config.keys()]:
        to_print += str(hp) + ": " + str(config[hp]) + "\t"
    print(to_print, flush=True)


def train_and_evaluation_transformer(train, early, val, early_stopping_kws, params):
    # extract validation data
    X_val, y_val = val

    # fit model
    model = train_transformer(train, early, early_stopping_kws, params)

    # validation prediction
    y_val_preds = model.predict(X_val)

    # validation evaluation
    cur_mse = mean_squared_error(y_val_preds, torch.FloatTensor(y_val))
    if np.isnan(cur_mse):
        print("ERROR")
        return np.inf
    return cur_mse


def train_and_evaluation_rnn(train, early, val, early_stopping_kws, params):
    # complexity restriction

    # extract vlidation data
    X_val, y_val, len_val = val

    # fit model
    model = train_rnn(train, early, early_stopping_kws, params)

    # validation prediction
    y_val_preds = model.predict(X_val, len_val, numpy=True)

    # validation evaluation
    cur_mse = mean_squared_error(y_val_preds, torch.FloatTensor(y_val))
    return cur_mse


def modify_config(params):
    config = params.copy()
    if model_name == 'lstm':
        if config['num_layers'] > 1:
            config['hidden_size'] = min(config['hidden_size'], 2 ** 9)

    if model_name == 'transformer':
        if data_name == 'tasmc':
            config['nhead'] = 1

    if data_name == 'proact':
        config['batch_size'] *= 2

    return config


def tuning():
    # get data
    data = get_data()
    X_train, y_train, len_train = data[0]
    X_early, y_early, len_early = data[1]
    X_val, y_val, len_val = data[2]

    # early stopping args
    early_stopping_kws = dict(patience=15, verbose=False, delta=0, path=model_path)

    # configurations
    param_dist = get_param_dist(model_name)
    param_sampler = ParameterSampler(param_dist, n_random_search, random_state=123)

    # initializations
    tuning_table = pd.DataFrame()
    if not os.path.exists(tuning_path):
        tuning_table = tuning_initialization(param_dist, tuning_path)
    # min_validation_mse = min_validation_mse = np.inf

    for i, params in enumerate(param_sampler):
        if os.path.exists(tuning_path):
            tuning_table = pd.read_csv(tuning_path)

        current = tuning_table.shape[0]

        if current == i:
            s = time.time()
            cur_mse = np.inf
            # headline
            plot_headline(i, n_random_search, params)

            # manipulation on the configuration (if actions are needed)
            params = modify_config(params)

            # train and validation evaluation
            if model_name == 'rnn' or model_name == 'gru' or model_name == 'lstm':
                cur_mse = train_and_evaluation_rnn((X_train, y_train, len_train),
                                                   (X_early, y_early, len_early),
                                                   (X_val, y_val, len_val),
                                                   early_stopping_kws, params)

            if model_name == 'transformer':
                cur_mse = train_and_evaluation_transformer((X_train, y_train), (X_early, y_early), (X_val, y_val),
                                                           early_stopping_kws, params)

            # save configuration
            params['validation_mse'] = cur_mse
            params['training_time'] = time.time() - s
            tuning_table = tuning_table.append(params, ignore_index=True)

            # save tuning table
            print(f"saving csv \'tuning_{permutation_number}\'.csv")
            tuning_table.to_csv(tuning_path, index=False)


def update_best_score(cur_mse, min_validation_mse):
    if cur_mse < min_validation_mse:
        min_validation_mse = cur_mse
        print(f"Validation decreased to {min_validation_mse}", flush=True)
    return min_validation_mse


def plot_headline(i, n_iter, params):
    print(f"---------------------------------------------------------")
    print(f"\t\t Random search iteration number {i + 1}/{n_iter} ")
    print(f"---------------------------------------------------------")
    print("configuration:\n")
    print_config(params)


def tuning_initialization(param_dist, tuning_path):
    tuning_table = pd.DataFrame(columns=[*param_dist.keys()] + ['training_time', 'validation_mse'])
    tuning_table.to_csv(tuning_path, index=False)
    return tuning_table


def train_transformer(train_data, early_data, early_stopping_kws, params):
    X_train, y_train = train_data
    X_early, y_early = early_data

    model = TransformerRegressorModel(n_features=X_train[0].shape[1],
                                      nhead=params['nhead'],
                                      use_cuda=False,
                                      dim_feedforward=params['dim_feedforward'],
                                      num_layers=params['num_layers'])

    model.compile(optimizer='adam', loss='mse', lr=params['lr'])

    model.fit((X_train, y_train),
              epochs=3000,
              early=(X_early, y_early),
              batch_size=int(params['batch_size']),
              early_stopping_kws=early_stopping_kws)
    return model


def train_rnn(train_data, early_data, early_stopping_kws, params):
    X_train, y_train, len_train = train_data
    X_early, y_early, len_early = early_data

    model = RNNRegressorModel(input_size=X_train[0].shape[1],
                              rnn_hidden_units=[params['hidden_size']] * params['num_layers'],
                              ff_hidden=[1],
                              rnn_type=params['rnn_type'],
                              learn_initial_state=params['learn_initial_state'],
                              activation='tanh',
                              use_cuda=False,
                              norm_type=params['norm_type'])

    model.compile(optimizer='adam', loss='mse', lr=params['lr'], weight_decay=params['weight_decay'])

    model.fit(X_train, y_train, len_train,
              validation_data=(X_early, y_early, len_early),
              early_stopping=early_stopping_kws,
              batch_size=int(params['batch_size']),
              epochs=3000,
              use_cuda=False)

    return model


if __name__ == '__main__':
    n_random_search = 30

    for permutation_number in np.arange(0, 60):
        # define paths
        data_name, d_code = 'proact', 0
        model_name, m_code = 'rnn', 1

        tuning_path = f'reuslts/{d_code}{m_code}0/tuning_{permutation_number}.csv'
        model_path = f'results/{d_code}{m_code}1/{permutation_number}.model'

        #       1/0         0/1/2/3                    0/1/2
        # ------------------------------------------------------------
        # 0 - PROACT        0 - transformer        0 - tuning
        # 1 - TASMC         1 - rnn                1 - models
        #                   2 - gru                2 - test
        #                   3 - lstm

        print(f"Permutation: {permutation_number}\n"
              f"Number of random search iterations: {n_random_search}\n"
              f"Model: {model_name}\n"
              f"Data: {data_name}\n"
              f"tuning path: {tuning_path}\n"
              f"model path: {model_path}")

        # hyper-parameters tuning
        tuning()

        print("Finished successfully")
