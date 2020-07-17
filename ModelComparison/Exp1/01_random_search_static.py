# sys.path.append('/gpfs0/boaz/users/benhada/home/Master/')
# sys.path.append('/gpfs0/boaz/users/benhada/home/Master/models')

import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from xgboost.callback import early_stop

from models.NNRegressor import NNRegressorModel


class Timer:
    __slots__ = ['current', 'elapsed_time']

    def __init__(self):
        self.current = 0
        self.elapsed_time = 0

    def on(self):
        self.current = time.time()

    def off(self):
        self.elapsed_time = time.time() - self.current

    def off_and_print(self):
        self.off()
        print(f'{self.elapsed_time:.3f} secs')

    def get_run_time(self):
        self.off()
        return self.elapsed_time


def load_data_for_train(number_iter, early_stopping=True):
    data_file = None
    splits_info = None
    if database == 'proact':
        data_file = '../f04_proact_flat.csv'
        splits_info = pickle.load(open('../f01_proact_split_info.pkl', 'rb'))
    if database == 'tasmc':
        data_file = 'data/tasmc_static/tasmc_flat.csv'
        splits_info = pickle.load(open('../f11_tasmc_split_info.pkl', 'rb'))

    X = pd.read_csv(data_file)

    X_train = X[X['ID_temp'].isin(splits_info[number_iter]['train_pats'])]
    X_train.reset_index(drop=True, inplace=True)
    y_train = X_train['y'].values
    X_train.drop('y', 1, inplace=True)

    X_early = get_validation_data(X, splits_info[number_iter]['early_pats'])
    X_early.reset_index(drop=True, inplace=True)
    y_early = X_early['y'].values
    X_early.drop('y', 1, inplace=True)

    X_val = get_validation_data(X, splits_info[number_iter]['val_pats'])
    X_val.reset_index(drop=True, inplace=True)
    y_val = X_val['y'].values
    X_val.drop('y', 1, inplace=True)

    if check_intersected_IDs(X_train, X_early):
        raise Exception("PROBLEM")
    if check_intersected_IDs(X_early, X_val):
        raise Exception("PROBLEM")
    if check_intersected_IDs(X_train, X_val):
        raise Exception("PROBLEM")

    if not early_stopping:
        X_train = pd.concat([X_train, X_early])
        y_train = np.concatenate([y_train, y_early])
        X_early, y_early = None, None

    return X_train, y_train, X_early, y_early, X_val, y_val


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


def plot_headline(config, i, n_iter):
    print(f"------------------------------------------------------------ ")
    print(f"        Random search iteration number {i + 1}/{n_iter}      ")
    print(f"------------------------------------------------------------ ")

    to_print = ''
    for hp in config:
        to_print += str(hp) + ": " + str(config[hp]) + "\t"
    print(to_print + '\n', flush=True)


def get_param_dist(model_name):
    param_dist = None
    if model_name == 'XGB':
        param_dist = get_xgb_param_dist()

    if model_name == 'RF':
        param_dist = get_rf_param_dist()

    if model_name == 'mlp':
        param_dist = get_mlp_param_dist()
    return param_dist


def get_xgb_param_dist():
    param_dist = {'max_depth': np.arange(2, 30),
                  'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
                  'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                  'gamma': [0, 0.25, 0.5, 1.0],
                  'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
                  'tree_method': ['auto']}
    return param_dist


def get_rf_param_dist():
    param_dist = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 300, 400, 500, 600, 700, 800, 1000]}
    return param_dist


def get_mlp_param_dist():
    param_dist = dict(n_hidden_layers=[1, 2, 3, 4, 5],
                      n_hidden_units=[16, 32, 64, 128, 256, 512, 1024],
                      activation=['relu', 'tanh', 'sigmoid'],
                      optimizer=['adam'],
                      batch_size=2 ** np.arange(5, 10),
                      lr=[0.0001, 0.001, 0.01],
                      regularizer_val=[0, 0.0001, 0.001, 0.01, 1.],
                      dropout=[0.0, 0.1, 0.2],
                      batch_norm=[True, False])
    return param_dist


def train_and_evaluate_rf(train, val, params):
    # get data
    X_train, y_train = train
    X_val, y_val = val

    # handle missing values
    X_train, X_val = complete_missing_values_before_training(X_train, X_val)

    # create model
    model = create_rf_model(params)

    model.fit(X_train.drop(['ID', 'ID_temp'], 1), y_train)

    # Validation evaluation
    y_val_preds = model.predict(X_val.drop(['ID', 'ID_temp'], 1))
    cur_mse = mean_squared_error(y_val_preds, y_val)
    return cur_mse


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


def modify_config(params, model_name):
    config = params.copy()
    # if model_name == 'MLP':
    #     if config['n_hidden_layers'] > 4:
    #         config['n_hidden_units'] = min(512, config['n_hidden_units'])
    return config


def tuning():
    # set early stopping only for XGB
    if model_name == 'XGB' or model_name == 'mlp':
        early_stopping = True
    else:
        early_stopping = False

    # load data for training and evaluation
    X_train, y_train, X_early, y_early, X_val, y_val = load_data_for_train(permutation, early_stopping)

    # get configurations distribution
    param_dist = get_param_dist(model_name)  # add options: NN and RF
    param_sampler = ParameterSampler(param_dist, n_random_search, random_state=123)

    # initialization
    min_validation_mse, tuning_table = initialize_best_score(param_dist)

    for i, params in enumerate(param_sampler):
        timer = Timer()
        timer.on()
        cur_mse = np.inf

        # modify config if necessary
        # params = modify_config(params, model_name)

        # plot iteration headline
        plot_headline(params, i, n_random_search)

        # XGB model
        if model_name == 'XGB':
            cur_mse = train_and_evaluate_xgb((X_train, y_train), (X_early, y_early), (X_val, y_val), params)

        # RF model
        if model_name == 'RF':
            # combine X_train and X_early
            cur_mse = train_and_evaluate_rf((X_train, y_train), (X_val, y_val), params)

        # MLP model
        if model_name == 'mlp':
            cur_mse = train_and_evaluate_mlp((X_train, y_train), (X_early, y_early), (X_val, y_val), params,
                                             model_file)

        # save best score
        min_validation_mse = update_best_score(cur_mse, min_validation_mse)

        # save configuration
        params['validation_mse'] = cur_mse
        params['training_time'] = timer.get_run_time()
        tuning_table = tuning_table.append(params, ignore_index=True)

        # Save tuning table
        tuning_table.to_csv(tuning_file, index=False)


def feature_filtering(x, y):
    from sklearn.feature_selection import f_regression

    X_complete = x.fillna(0)

    f_scores, p_vals = f_regression(X_complete.iloc[:, 2:], y)

    feature_importances = pd.DataFrame(dict(feature=X_complete.iloc[:, 2:].columns.tolist(),
                                            p_vals=p_vals,
                                            f_scores=f_scores))

    percentile = np.percentile(feature_importances.f_scores, 30)
    feature_importances = feature_importances[feature_importances.f_scores > percentile]
    return feature_importances.feature.tolist()


def initialize_best_score(param_dist):
    min_validation_mse = np.inf
    tuning_table = pd.DataFrame(columns=[*param_dist.keys()] + ['training_time', 'validation_mse'])
    return min_validation_mse, tuning_table


def update_best_score(cur_mse, min_validation_mse):
    if cur_mse < min_validation_mse:
        min_validation_mse = cur_mse
        print(f"Validation decreased to {min_validation_mse:.3f}\n", flush=True)
    return min_validation_mse


def train_and_evaluate_xgb(train, early, val, params):
    X_train, y_train = train
    X_early, y_early = early
    X_val, y_val = val

    # create model
    model = create_xgb_model(params)

    # drop skew
    for col in X_train.columns:
        if 'skew' in col:
            X_train.drop(col, 1, inplace=True)
            X_early.drop(col, 1, inplace=True)
            X_val.drop(col, 1, inplace=True)

    # Fit model using early stopping
    early = early_stop(stopping_rounds=30, maximize=False)
    model.fit(X_train.drop(['ID', 'ID_temp'], 1), y_train,
              eval_set=[(X_train.drop(['ID', 'ID_temp'], 1), y_train),
                        (X_early.drop(['ID', 'ID_temp'], 1), y_early)],
              callbacks=[early])
    # Validation evaluation
    y_val_preds = model.predict(X_val.drop(['ID', 'ID_temp'], 1))
    cur_mse = mean_squared_error(y_val_preds, y_val)
    return cur_mse


def check_intersected_IDs(x_train, x_val):
    obs_1 = x_train.ID_temp.unique()
    obs_2 = x_val.ID_temp.unique()
    if np.intersect1d(obs_1, obs_2).shape[0] > 0:
        return True
    else:
        return False


def scaling(x_train, x_val, x_early=None):
    scaler = MinMaxScaler(feature_range=(0.5, 1))
    x_train_scaled = scaler.fit_transform(x_train.drop(['ID', 'ID_temp'], 1))
    x_val_scaled = scaler.transform(x_val.drop(['ID', 'ID_temp'], 1))
    if x_early is not None:
        x_early_scaled = scaler.transform(x_early.drop(['ID', 'ID_temp'], 1))
    else:
        x_early_scaled = None
    return x_train_scaled, x_val_scaled, x_early_scaled


def shuffle_data(X_train, y_train):
    X_train['y'] = y_train
    new_X_train = X_train.sample(frac=1, random_state=123)
    new_y_train = X_train['y'].values
    new_X_train.drop('y', 1, inplace=True)
    return new_X_train, new_y_train


def train_and_fit_mlp(train, val, params, model_path):
    # set number of threads (for cluster)
    torch.set_num_threads(1)
    X_train, y_train = train
    X_val, y_val = val

    model = NNRegressorModel(X_train.shape[1],
                             [params['n_hidden_units']] * params['n_hidden_layers'],
                             activation=params['activation'],
                             dropout=params['dropout'],
                             batch_norm=params['batch_norm'],
                             use_cuda=True)

    model.compile(optimizer=params['optimizer'],
                  learning_rate=params['lr'],
                  l1=params['regularizer_val'])

    model.fit(X_train, y_train,
              validation_data=[X_val, y_val],
              epochs=3000,
              batch_size=int(params['batch_size']),
              model_path=model_path)
    return model


def train_and_evaluate_mlp(train, early, val, params, model_path):
    # get_data
    X_train, y_train = train
    X_early, y_early = early
    X_val, y_val = val

    if check_intersected_IDs(X_train, X_val):
        raise Exception("Intersected observations")
    if check_intersected_IDs(X_train, X_early):
        raise Exception("Intersected observations")
    if check_intersected_IDs(X_early, X_val):
        raise Exception("Intersected observations")

    # handle missing values left
    X_train, (X_early, X_val) = complete_missing_values_before_training(X_train, [X_early, X_val])

    # # shuffle X_train
    # X_train, y_train = shuffle_data(X_train, y_train)

    # scaling
    X_train, X_val, X_early = scaling(X_train, X_val, X_early)

    # create model
    model = train_and_fit_mlp((X_train, y_train), (X_early, y_early), params, model_path)

    # # Fit model using early stopping
    # model.fit(X_train, y_train)

    # Validation evaluation
    y_val_preds = model.predict(X_val)
    cur_mse = mean_squared_error(y_val_preds, y_val)

    return cur_mse


def create_xgb_model(params):
    model = XGBRegressor(max_depth=params['max_depth'],
                         learning_rate=params['learning_rate'],
                         subsample=params['subsample'],
                         colsample_bytree=params['colsample_bytree'],
                         colsample_bylevel=params['colsample_bylevel'],
                         min_child_weight=params['min_child_weight'],
                         gamma=params['gamma'],
                         reg_lambda=params['reg_lambda'],
                         n_estimators=31,  # TODO
                         objective='reg:squarederror',
                         tree_method=params['tree_method'])
    return model


def create_rf_model(params):
    model = RandomForestRegressor(n_estimators=2,  # params['n_estimators'],
                                  max_depth=params['max_depth'],
                                  max_features=params['max_features'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  min_samples_split=params['min_samples_split'],
                                  bootstrap=params['bootstrap'])
    return model


if __name__ == '__main__':
    d_code = dict(proact=0, tasmc=1)
    m_code = dict(mlp=5)

    permutation = 10
    database = "tasmc"  # 'tasmc' or 'proact
    model_name = "mlp"  # MLP, RF, XGB

    print("Permutation: ", permutation)
    print("Database: ", database)
    print("Model: ", model_name)

    n_random_search = 30
    tuning_file = f'results/{d_code[database]}{m_code[model_name]}0/{permutation}.csv'
    model_file = f'results/{d_code[database]}{m_code[model_name]}1/{permutation}.pt'

    # main
    tuning()
