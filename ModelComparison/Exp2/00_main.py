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


def get_partial_x(x, y, idx_map, pats, max_observations, random_state=123):
    random.seed(random_state)
    indices = []
    for ID in pats:
        indices += random.sample(idx_map[ID], k=min(len(idx_map[ID]), max_observations))
    indices.sort()
    return [x[i] for i in indices], [y[i] for i in indices]


def get_sequences_lengths(x):
    return torch.FloatTensor([x_i.shape[0] for x_i in x]).reshape(-1)


def get_tasmc_test(trainset):
    if trainset == 'proact':
        temporal_data_file = 'data/tasmc_temporal/tasmc_temporal_scaled_by_proact.pkl'
    else:
        temporal_data_file = 'data/tasmc_temporal/tasmc_temporal.pkl'
    splits_info_file = '../f11_tasmc_split_info.pkl'
    X, y, l, idx_map = pickle.load(open(temporal_data_file, 'rb'))
    splits_info = pickle.load(open(splits_info_file, 'rb'))
    return get_partial_x(X, y, idx_map, splits_info[permutation]['test_pats'], np.inf)


def get_data(trainset):
    if trainset == 'tasmc':
        temporal_data_file = 'data/tasmc_temporal/tasmc_temporal.pkl'
        splits_info_file = '../f11_tasmc_split_info.pkl'
    else:
        temporal_data_file = 'data/proact_temporal/proact_temporal.pkl'
        splits_info_file = '../f01_proact_split_info.pkl'

    # load data and split info
    X, y, l, idx_map = pickle.load(open(temporal_data_file, 'rb'))
    splits_info = pickle.load(open(splits_info_file, 'rb'))

    # extract X and y
    train_pats = list(splits_info[permutation]['train_pats']) + list(splits_info[permutation]['early_pats'])
    X_train, y_train = get_partial_x(X, y, idx_map, train_pats, np.inf)
    X_early, y_early = get_partial_x(X, y, idx_map, splits_info[permutation]['val_pats'], 4)

    # get X, y test
    X_test, y_test = get_tasmc_test(trainset)

    # calculate lengths
    len_train = get_sequences_lengths(X_train)
    len_early = get_sequences_lengths(X_early)
    len_test = get_sequences_lengths(X_test)

    return (X_train, y_train, len_train), (X_early, y_early, len_early), (X_test, y_test, len_test)


def train_transformer(train_data, early_data, early_stopping_kws, config):
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


def fit_predict_transformer(train, early, val, early_stopping_kws, config):
    # extract validation data
    X_val, y_val = val

    # fit model
    model = train_transformer(train, early, early_stopping_kws, config)

    # validation prediction
    return model.predict(X_val)


def fit_temporal(trainset, config, model_path):
    # get data
    (X_train, y_train, len_train), (X_early, y_early, len_early), (X_test, y_test, len_test) = get_data(trainset)

    # early stopping args
    early_stopping_kws = dict(patience=15, verbose=False, delta=0, path=model_path)

    # get configuration
    test_predictions = None
    if modelnm in ['rnn', 'gru', 'lstm']:
        test_predictions = fit_predict_rnn((X_train, y_train, len_train),
                                           (X_early, y_early, len_early),
                                           (X_test, y_test, len_test),
                                           early_stopping_kws,
                                           config)

    if modelnm == 'transformer':
        test_predictions = fit_predict_transformer((X_train, y_train),
                                                   (X_early, y_early),
                                                   (X_test, y_test),
                                                   early_stopping_kws,
                                                   config)
    print("RMSE: ", round(mean_squared_error(y_test, test_predictions) ** 0.5, 2))
    return test_predictions


def train_rnn(train_data, early_data, early_stopping_kws, config):
    X_train, y_train, len_train = train_data
    X_early, y_early, len_early = early_data

    model = RNNRegressorModel(input_size=X_train[0].shape[1],
                              rnn_hidden_units=[config['hidden_size']] * config['num_layers'],
                              ff_hidden=[1],
                              rnn_type=modelnm,
                              learn_initial_state=config['learn_initial_state'],
                              activation='tanh',
                              use_cuda=False,
                              norm_type=config['norm_type'])

    model.compile(optimizer='adam', loss='mse', lr=config['lr'], weight_decay=config['weight_decay'])

    model.fit(X_train, y_train, len_train,
              validation_data=(X_early, y_early, len_early),
              early_stopping=early_stopping_kws,
              batch_size=int(config['batch_size']),
              epochs=3000,
              use_cuda=False)

    return model


def fit_predict_rnn(train, early, test, early_stopping_kws, config):
    # extract vlidation data
    X_test, y_test, len_test = test

    # fit model
    model = train_rnn(train, early, early_stopping_kws, config)

    # validation prediction
    return model.predict(X_test, len_test, numpy=True)


def use_early_stopping() -> bool:
    return True if modelnm == 'xgb' or modelnm == 'mlp' else False


def check_intersected_identifiers(x_train, x_val):
    obs_1 = list(map(str, x_train.ID_temp.unique()))
    obs_2 = list(map(str, x_val.ID_temp.unique()))
    if np.intersect1d(obs_1, obs_2).shape[0] > 0:
        return True
    else:
        return False


def get_validation_data(x, patients, random_state=32):
    np.random.seed(random_state)
    indices = []
    for ID in patients:
        ID_indices = x[x['ID_temp'] == ID].index.tolist()
        chosen = random.sample(ID_indices, k=min(4, len(ID_indices)))
        indices += chosen
    X_early = x.loc[indices].copy()
    return X_early


def load_flattened_data(trainset, early_stopping=True):
    test_data_file = 'data/tasmc_static/tasmc_flat.csv'
    if trainset == 'proact':
        train_data_file = 'data/proact_static/proact_flat.csv'
        train_splits_info = pickle.load(open('../f01_proact_split_info.pkl', 'rb'))
    else:
        train_data_file = 'data/tasmc_static/tasmc_flat.csv'
        train_splits_info = pickle.load(open('../f11_tasmc_split_info.pkl', 'rb'))

    test_splits_info = pickle.load(open('../f11_tasmc_split_info.pkl', 'rb'))

    X_for_train = pd.read_csv(train_data_file)
    X_for_test = pd.read_csv(test_data_file)

    train_pats = list(train_splits_info[permutation]['train_pats']) + list(train_splits_info[permutation]['early_pats'])

    X_train = X_for_train[X_for_train['ID_temp'].isin(train_pats)]
    X_train.reset_index(drop=True, inplace=True)
    y_train = X_train['y'].values
    X_train.drop('y', 1, inplace=True)

    X_early = get_validation_data(X_for_train, train_splits_info[permutation]['val_pats'])
    X_early.reset_index(drop=True, inplace=True)
    y_early = X_early['y'].values
    X_early.drop('y', 1, inplace=True)

    X_test = X_for_test[X_for_test['ID_temp'].isin(test_splits_info[permutation]['test_pats'])]
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


def fit_static(trianset, config, model_path):
    # set early stopping for XGB and mlp
    early_stopping = use_early_stopping()

    X_train, y_train, X_early, y_early, X_test, y_test = load_flattened_data(trianset, early_stopping)

    test_predictions = None
    if modelnm == 'mlp':
        test_predictions = fit_predict_mlp((X_train, y_train), (X_early, y_early), (X_test, None), config, model_path)

    print("RMSE: :", mean_squared_error(test_predictions, y_test)**0.5)

    return test_predictions


def save_predictions(preds, path):
    print("saving predictions in ", path, "...", end='')
    pd.DataFrame(preds, columns=['y']).to_csv(path, index=False)
    print("saved")

#
# def complete_missing_values_before_training(x_train, val_datasets):
#     from sklearn.impute import SimpleImputer
#
#     categorical = []
#     numeric = []
#     for col in x_train.columns.drop(['ID', 'ID_temp']):
#         if x_train[col].nunique() < 3:
#             categorical.append(col)
#         else:
#             numeric.append(col)
#
#     x_train = x_train.replace(-np.inf, np.nan)
#     x_train = x_train.replace(np.inf, np.nan)
#     for i in range(len(val_datasets)):
#         val_datasets[i] = val_datasets[i].replace(np.inf, np.nan)
#         val_datasets[i] = val_datasets[i].replace(-np.inf, np.nan)
#
#     numeric_imputer = SimpleImputer(strategy='mean')
#     categorical_imputer = SimpleImputer(strategy='most_frequent')
#
#     x_train[numeric] = numeric_imputer.fit_transform(x_train[numeric])
#     x_train[categorical] = categorical_imputer.fit_transform(x_train[categorical])
#
#     for i in range(len(val_datasets)):
#         val_datasets[i][numeric] = numeric_imputer.transform(val_datasets[i][numeric])
#         val_datasets[i][categorical] = categorical_imputer.transform(val_datasets[i][categorical])
#
#     return x_train, val_datasets


def scaling(x_train, x_val, x_early=None):
    scaler = MinMaxScaler(feature_range=(0.5, 1))
    x_train_scaled = scaler.fit_transform(x_train.drop(['ID', 'ID_temp'], 1))
    x_val_scaled = scaler.transform(x_val.drop(['ID', 'ID_temp'], 1))
    if x_early is not None:
        x_early_scaled = scaler.transform(x_early.drop(['ID', 'ID_temp'], 1))
    else:
        x_early_scaled = None
    return x_train_scaled, x_val_scaled, x_early_scaled


def fit_predict_mlp(train, early, test, config, model_path):
    # get_data
    x_train, y_train = train
    x_early, y_early = early
    x_test, _ = test

    if check_intersected_identifiers(x_train, x_test):
        raise Exception("Intersected observations")
    if check_intersected_identifiers(x_train, x_early):
        raise Exception("Intersected observations")
    if check_intersected_identifiers(x_early, x_test):
        raise Exception("Intersected observations")

    # handle missing values left
    # x_train, (x_early, X_val) = complete_missing_values_before_training(x_train, [x_early, x_test])

    # scaling
    x_train, x_test, x_early = scaling(x_train, x_test, x_early)

    # fit model
    model = train_and_fit_mlp((x_train, y_train), (x_early, y_early), config, model_path)

    # test predictions
    return model.predict(x_test)


def train_and_fit_mlp(train, val, config, model_path):
    # set number of threads (for cluster)
    x_train, y_train = train
    x_val, y_val = val

    model = NNRegressorModel(x_train.shape[1],
                             [config['n_hidden_units']] * config['n_hidden_layers'],
                             activation=config['activation'],
                             dropout=config['dropout'],
                             batch_norm=config['batch_norm'],
                             use_cuda=False)

    model.compile(optimizer='adam',
                  learning_rate=config['lr'],
                  l1=config['regularizer_val'])

    model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              epochs=3000,
              batch_size=int(config['batch_size']),
              model_path=model_path)
    return model


def d_code(x):
    return dict(proact=0, tasmc=1)[x]


def m_code(x):
    return dict(transformer=0, rnn=1, gru=2, mlp=5)[x]


def get_config(trainset):
    return pickle.load(open("../Exp1/f18_best_configs.pkl", "rb"))[trainset][modelnm][permutation]


def is_static_model():
    return modelnm in ['mlp', 'rf', 'xgb']


def fit_predict(trainset):
    model_path = f'results/{d_code(trainset)}{m_code(modelnm)}1/{permutation}.model'
    preds_path = f'results/{d_code(trainset)}{m_code(modelnm)}2/{permutation}.csv'

    # config = get_config(trainset)
    config = get_config('proact')
    # config['norm_type'] = 'batch'
    print("Config: ", config)

    if is_static_model():
        preds = fit_static(trainset, config, model_path)
    else:
        preds = fit_temporal(trainset, config, model_path)
    save_predictions(preds, preds_path)


if __name__ == '__main__':
    modelnm = 'mlp'
    permutation = 0
    fit_predict('tasmc')
