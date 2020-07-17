"""Test run for temporal models: RNN, GRU, LSTM, Transformer"""
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from models.NNRegressor import NNRegressorModel
from models.RNNRegressor import RNNRegressorModel
from models.TransformerRegressor import TransformerRegressorModel

torch.set_num_threads(1)


def get_partial_x(X, y, idx_map, pats, max_observations, random_state=123):
    random.seed(random_state)
    indices = []
    for ID in pats:
        indices += random.sample(idx_map[ID], k=min(len(idx_map[ID]), max_observations))
    indices.sort()
    return [X[i] for i in indices], [y[i] for i in indices]


def get_data():
    temporal_data_file = 'data/tasmc_temporal/tasmc_temporal.pkl' if data_name == 'tasmc' \
        else 'data/proact_temporal/proact_temporal.pkl'
    splits_info_file = '../f11_tasmc_split_info.pkl' if data_name == 'tasmc' else '../f01_proact_split_info.pkl'

    # load data and split info
    X, y, l, idx_map = pickle.load(open(temporal_data_file, 'rb'))
    splits_info = pickle.load(open(splits_info_file, 'rb'))

    # extract X and y
    train_pats = list(splits_info[permutation]['train_pats']) + list(splits_info[permutation]['early_pats'])
    X_train, y_train = get_partial_x(X, y, idx_map, train_pats, np.inf)
    X_early, y_early = get_partial_x(X, y, idx_map, splits_info[permutation]['val_pats'], 4)
    X_test, y_test = get_partial_x(X, y, idx_map, splits_info[permutation]['test_pats'], np.inf)

    # calculate lengths
    len_train = get_sequences_lengths(X_train)
    len_early = get_sequences_lengths(X_early)
    len_test = get_sequences_lengths(X_test)

    return (X_train, y_train, len_train), (X_early, y_early, len_early), (X_test, y_test, len_test)


def get_sequences_lengths(x):
    return torch.FloatTensor([x_i.shape[0] for x_i in x]).reshape(-1)


def plot_settings():
    print("Model: ", model_name)
    print("Database: ", data_name)


def train_and_fit_mlp(train, val):
    # set number of threads (for cluster)
    X_train, y_train = train
    X_val, y_val = val

    model = NNRegressorModel(X_train.shape[1],
                             [config['n_hidden_units']] * config['n_hidden_layers'],
                             activation=config['activation'],
                             dropout=config['dropout'],
                             batch_norm=config['batch_norm'],
                             use_cuda=False)

    model.compile(optimizer='adam',
                  learning_rate=config['lr'],
                  l1=config['regularizer_val'])

    model.fit(X_train, y_train,
              validation_data=[X_val, y_val],
              epochs=3000,
              batch_size=int(config['batch_size']),
              model_path=model_path)
    return model


def check_intersected_identifiers(x_train, x_val):
    obs_1 = x_train.ID_temp.unique()
    obs_2 = x_val.ID_temp.unique()
    if np.intersect1d(obs_1, obs_2).shape[0] > 0:
        return True
    else:
        return False


def get_validation_data(x, patients, random_state=32):
    """
    for each patient, sample maximum 4 observations

    Parameters
    ----------
    x
        pd.DataFrame - flattened database
    patients
        list - patients list
    random_state
        int
    Returns
    -------
        pd.DataFrame - X validation (includes y)
    """

    np.random.seed(random_state)
    indices = []
    for ID in patients:
        ID_indices = x[x['ID_temp'] == ID].index.tolist()
        chosen = random.sample(ID_indices, k=min(4, len(ID_indices)))
        indices += chosen
    X_early = x.loc[indices].copy()
    return X_early


def load_flattened_data(early_stopping=True):
    data_file = None
    splits_info = None
    if data_name == 'proact':
        data_file = '../f04_proact_flat.csv'
        splits_info = pickle.load(open('../f01_proact_split_info.pkl', 'rb'))
    if data_name == 'tasmc':
        data_file = 'data/tasmc_static/tasmc_flat_processed.csv'
        splits_info = pickle.load(open('../f11_tasmc_split_info.pkl', 'rb'))

    X = pd.read_csv(data_file)

    train_pats = list(splits_info[permutation]['train_pats']) + list(splits_info[permutation]['early_pats'])

    X_train = X[X['ID_temp'].isin(train_pats)]
    X_train.reset_index(drop=True, inplace=True)
    y_train = X_train['y'].values
    X_train.drop('y', 1, inplace=True)

    X_early = get_validation_data(X, splits_info[permutation]['val_pats'])
    X_early.reset_index(drop=True, inplace=True)
    y_early = X_early['y'].values
    X_early.drop('y', 1, inplace=True)

    X_test = X[X['ID_temp'].isin(splits_info[permutation]['test_pats'])]
    X_test.reset_index(drop=True, inplace=True)
    y_test = X_test['y'].values
    X_test.drop('y', 1, inplace=True)

    if check_intersected_identifiers(X_train, X_early):
        raise Exception("PROBLEM")
    if check_intersected_identifiers(X_early, X_test):
        raise Exception("PROBLEM")
    if check_intersected_identifiers(X_train, X_test):
        raise Exception("PROBLEM")

    if not early_stopping:
        X_train = pd.concat([X_train, X_early])
        y_train = np.concatenate([y_train, y_early])
        X_early, y_early = [None] * 2

    return X_train, y_train, X_early, y_early, X_test, y_test


def static_test_run():
    # set early stopping for XGB and mlp
    early_stopping = use_early_stopping()

    X_train, y_train, X_early, y_early, X_test, y_test = load_flattened_data(early_stopping)

    # MLP model
    test_predictions = None
    if model_name == 'mlp':
        test_predictions = fit_predict_mlp((X_train, y_train), (X_early, y_early), (X_test, None))

    if test_predictions is not None:
        print("RMSE: ", round(mean_squared_error(test_predictions, y_test) ** .5, 2))
        save_predictions(test_predictions)
    else:
        raise RuntimeError


def save_predictions(test_predictions):
    print("saving predictions in ", predictions_path, "...", end='')
    pd.DataFrame(test_predictions, columns=['y']).to_csv(predictions_path, index=False)
    print("saved")


def use_early_stopping() -> bool:
    return True if model_name == 'xgb' or model_name == 'mlp' else False


def complete_missing_values_before_training(x_train, val_datasets):
    from sklearn.impute import SimpleImputer

    categorical = []
    numeric = []
    for col in x_train.columns.drop(['ID', 'ID_temp']):
        if x_train[col].nunique() < 3:
            categorical.append(col)
        else:
            numeric.append(col)

    x_train = x_train.replace(-np.inf, np.nan)
    x_train = x_train.replace(np.inf, np.nan)
    for i in range(len(val_datasets)):
        val_datasets[i] = val_datasets[i].replace(np.inf, np.nan)
        val_datasets[i] = val_datasets[i].replace(-np.inf, np.nan)

    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    x_train[numeric] = numeric_imputer.fit_transform(x_train[numeric])
    x_train[categorical] = categorical_imputer.fit_transform(x_train[categorical])

    for i in range(len(val_datasets)):
        val_datasets[i][numeric] = numeric_imputer.transform(val_datasets[i][numeric])
        val_datasets[i][categorical] = categorical_imputer.transform(val_datasets[i][categorical])

    return x_train, val_datasets


def fit_predict_mlp(train, early, test):
    # get_data
    X_train, y_train = train
    X_early, y_early = early
    X_test, _ = test

    if check_intersected_identifiers(X_train, X_test):
        raise Exception("Intersected observations")
    if check_intersected_identifiers(X_train, X_early):
        raise Exception("Intersected observations")
    if check_intersected_identifiers(X_early, X_test):
        raise Exception("Intersected observations")

    # handle missing values left
    X_train, (X_early, X_val) = complete_missing_values_before_training(X_train, [X_early, X_test])

    # scaling
    X_train, X_test, X_early = scaling(X_train, X_test, X_early)

    # fit model
    model = train_and_fit_mlp((X_train, y_train), (X_early, y_early))

    # test predictions
    return model.predict(X_test)


def scaling(x_train, x_val, x_early=None):
    scaler = MinMaxScaler(feature_range=(0.5, 1))
    x_train_scaled = scaler.fit_transform(x_train.drop(['ID', 'ID_temp'], 1))
    x_val_scaled = scaler.transform(x_val.drop(['ID', 'ID_temp'], 1))
    if x_early is not None:
        x_early_scaled = scaler.transform(x_early.drop(['ID', 'ID_temp'], 1))
    else:
        x_early_scaled = None
    return x_train_scaled, x_val_scaled, x_early_scaled


def train_rnn(train_data, early_data, early_stopping_kws):
    X_train, y_train, len_train = train_data
    X_early, y_early, len_early = early_data

    model = RNNRegressorModel(input_size=X_train[0].shape[1],
                              rnn_hidden_units=[config['hidden_size']] * config['num_layers'],
                              ff_hidden=[1],
                              rnn_type=model_name,
                              learn_initial_state=config['learn_initial_state'],
                              activation='tanh',
                              use_cuda=False,
                              norm_type=config['norm_type'])

    model.compile(optimizer='adam', loss='mse', lr=config['lr'], weight_decay=config['weight_decay'])

    model.fit(X_train, y_train, len_train,
              validation_data=(X_early, y_early, len_early),
              early_stopping=early_stopping_kws,
              batch_size=int(config['batch_size']),
              epochs=3000,  # TODO
              use_cuda=False)

    return model


def fit_predict_rnn(train, early, test, early_stopping_kws):
    # extract vlidation data
    X_test, y_test, len_test = test

    # fit model
    model = train_rnn(train, early, early_stopping_kws)

    # validation prediction
    test_predictions = model.predict(X_test, len_test, numpy=True)
    print("RMSE: ", round(mean_squared_error(test_predictions, y_test) ** .5, 2))
    save_predictions(test_predictions)


def temporal_test_run():
    # get data
    (X_train, y_train, len_train), (X_early, y_early, len_early), (X_test, y_test, len_test) = get_data()

    # early stopping args
    early_stopping_kws = dict(patience=15, verbose=False, delta=0, path=model_path)

    # get configuration
    # train and validation evaluation
    if model_name == 'rnn' or model_name == 'gru' or model_name == 'lstm':
        fit_predict_rnn((X_train, y_train, len_train),
                        (X_early, y_early, len_early),
                        (X_test, y_test, len_test),
                        early_stopping_kws)

    if model_name == 'transformer':
        fit_predict_transformer((X_train, y_train),
                                (X_early, y_early),
                                (X_test, y_test),
                                early_stopping_kws)


def train_transformer(train_data, early_data, early_stopping_kws):
    X_train, y_train = train_data
    X_early, y_early = early_data

    model = TransformerRegressorModel(n_features=int(X_train[0].shape[1]),
                                      nhead=int(config['nhead']),
                                      use_cuda=False,
                                      dim_feedforward=int(config['dim_feedforward']),
                                      num_layers=int(config['num_layers']))

    model.compile(optimizer='adam', loss='mse', lr=config['lr'])

    model.fit((X_train, y_train),
              epochs=3000,
              early=(X_early, y_early),
              batch_size=int(config['batch_size']),
              early_stopping_kws=early_stopping_kws)
    return model


def fit_predict_transformer(train, early, val, early_stopping_kws):
    # extract validation data
    X_val, y_val = val

    # fit model
    model = train_transformer(train, early, early_stopping_kws)

    # validation prediction
    test_predictions = model.predict(X_val)

    print("RMSE: ", round(mean_squared_error(test_predictions, y_val) ** .5, 2))

    # validation evaluation
    if test_predictions is not None:
        save_predictions(test_predictions)
    else:
        raise RuntimeError


def run_test():
    if model_name in ['mlp', 'rf', 'xgb']:
        static_test_run()

    if model_name in ['transformer', 'rnn', 'gru', 'lstm']:
        temporal_test_run()


def get_config():
    return pickle.load(open("../Exp1/f18_best_configs.pkl", "rb"))[data_name][model_name][permutation]


def d_code(x):
    return dict(proact=0, tasmc=1)[x]


def m_code(x):
    return dict(transformer=0, rnn=1, gru=2, mlp=5)[x]


if __name__ == '__main__':

    # settings
    model_name = 'transformer'
    data_name = 'proact'
    plot_settings()

    for permutation in range(33, 34):
        print("Permutation: ", permutation)
        # predictions_path = f'results/{d_code(data_name)}{m_code(model_name)}2/{permutation}.csv'
        model_path = f'results/{d_code(data_name)}{m_code(model_name)}1/{permutation}.pt'
        predictions_path = f'garb/{d_code(data_name)}{m_code(model_name)}2/{permutation}.csv'
        print("Save predictions to: ", predictions_path)
        config = get_config()
        config['batch_size'] /= 2
        print("Configuration: \n", config)
        run_test()
