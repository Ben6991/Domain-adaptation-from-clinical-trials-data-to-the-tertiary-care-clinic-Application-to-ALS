import os
import pickle
import random

import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
from tqdm import tqdm


def get_time_range(t):
    period = 4
    if t <= 6 * (365 / 12):
        period = 1
    elif t <= 12 * (365 / 12):
        period = 2
    elif t <= 18 * (365 / 12):
        period = 3
    return period


def get_prediction(x, test_indices, predictions, datanm):
    identifier, i = x.split('_')
    i = int(i)
    j = None
    # if datanm == 'tasmc':
    j = np.argwhere(np.array(test_indices) == idx_map_t[identifier][i]).item()
    # elif datanm == 'proact':
    #     j = np.argwhere(np.array(test_indices) == idx_map_p[int(identifier)][i]).item()
    return predictions[j].item()


def get_split_info(datanm):
    # if datanm == 'tasmc':
    return pickle.load(open("../f11_tasmc_split_info.pkl", "rb"))
    # elif datanm == 'proact':
    #     return pickle.load(open("../f01_proact_split_info.pkl", "rb"))


def get_x_test(datanm, permutation):
    split_info = get_split_info(datanm)
    columns = ['ID', 'ID_temp', 'n_visits', 'ALSFRS_last', 't_pred', 't_range', 'y']
    # if datanm == 'tasmc':
    return X_flat_t.loc[X_flat_t['ID_temp'].isin(list(split_info[permutation]['test_pats'])), columns]
    # elif datanm == 'proact':
    #     return X_flat_p.loc[X_flat_p['ID_temp'].isin(list(split_info[permutation]['test_pats'])), columns]


def get_test_indices(datanm, permutation):
    split_info = get_split_info(datanm)
    idx_map = dict()
    # if datanm == 'tasmc':
    idx_map = idx_map_t.copy()
    # elif datanm == 'proact':
    #     idx_map = idx_map_p
    test_indices = []
    for ID in split_info[permutation]['test_pats']:
        test_indices += idx_map[ID]
    test_indices.sort()
    return test_indices


def bootstrapping_analysis(res):
    random.seed(1)
    bootstrapping = {t: dict(rmse=[], mae=[]) for t in res['t_period'].unique()}
    for i in range(1000):
        for t in res['t_period'].unique():
            samp = resample(res[res['t_period'] == t], replace=True, n_samples=800)
            rmse = mean_squared_error(samp['y'], samp['pred']) ** 0.5
            mae = mean_absolute_error(samp['y'], samp['pred'])
            bootstrapping[t]['rmse'].append(rmse)
            bootstrapping[t]['mae'].append(mae)

    bootstrapping[99] = dict(rmse=[], mae=[])
    for i in range(1000):
        samp = resample(res, replace=True, n_samples=800)
        rmse = mean_squared_error(samp['y'], samp['pred']) ** 0.5
        mae = mean_absolute_error(samp['y'], samp['pred'])
        bootstrapping[99]['rmse'].append(rmse)
        bootstrapping[99]['mae'].append(mae)

    return bootstrapping


def get_results_temporal_model(datanm, modelnm):
    """ pd.DataFrame with columns: [ID, ID_temp, t_pred, n_visits, ALSFRS_last, permutation, y, pred]
    """
    res = pd.DataFrame()
    print("runs over 60 permutation")
    for permutation in tqdm(range(60)):
        if predictions_exist(datanm, modelnm, permutation):
            x_test = get_x_test(datanm, permutation)
            preds = get_predictions(x_test, datanm, modelnm, permutation)
            x_test['pred'] = preds
            x_test['permut'] = permutation
            print(f"{permutation}: {mean_squared_error(x_test['y'], x_test['pred']) ** 0.5:.2f}")
            res = pd.concat([res, x_test])
        else:
            print(f">> data {datanm}, model {modelnm} permutation {permutation} does not exist!")
    return res


def is_static_model(modelnm):
    return modelnm in ['rf', 'xgb', 'mlp']


def get_predictions(x_test, datanm, modelnm, permutation):
    test_predictions = load_predictions(datanm, modelnm, permutation)
    if is_static_model(modelnm):
        return test_predictions
    else:
        test_indices = get_test_indices(datanm, permutation)
        return x_test['ID'].apply(lambda x: get_prediction(x, test_indices, test_predictions, datanm))


def load_predictions(datanm, modelnm, permutation):
    file_path = f'results/{d_code[datanm]}{m_code[modelnm]}2/{permutation}.csv'
    return pd.read_csv(file_path).values


def predictions_exist(datanm, modelnm, permutation):
    file_path = f'results/{d_code[datanm]}{m_code[modelnm]}2/{permutation}.csv'
    return os.path.exists(file_path)


def get_results_static_model(datanm, modelnm):
    """ pd.DataFrame with columns: [ID, ID_temp, t_pred, n_visits, ALSFRS_last, permutation, y, pred]
    """
    res = pd.DataFrame()
    print("run over 60 permutation")
    for permutation in tqdm(range(60)):
        if predictions_exist(datanm, modelnm, permutation):
            x_test = get_x_test(datanm, permutation)
            preds = get_predictions(x_test, datanm, modelnm, permutation)
            x_test['pred'] = preds
            x_test['permut'] = permutation
            rmse_i = mean_squared_error(x_test['y'], x_test['pred']) ** 0.5
            print(f"{permutation}: {rmse_i:.2f}")
            if rmse_i < 15:
                res = pd.concat([res, x_test])
            else:
                print("X")
        else:
            print(f">> data {datanm}, model {modelnm} permutation {permutation} does not exist!")
    return res


def plot_bootstrapping_results(boot):
    ts = [*boot]
    ts.sort()
    for t in ts:
        print("t: ", t)
        print("RMSE:", round(np.mean(boot[t]['rmse']), 2), round(np.std(boot[t]['rmse']), 2))
        print("MAE:", round(np.mean(boot[t]['mae']), 2), round(np.std(boot[t]['mae']), 2))
        print("---")


def run(data_name, model_name):
    print("\n--------------------------------------------")
    print("Data: ", data_name, " Model: ", model_name)
    print("--------------------------------------------\n")
    if is_static_model(model_name):
        res_all = get_results_static_model(data_name, model_name)
    else:
        res_all = get_results_temporal_model(data_name, model_name)
    res_all['t_period'] = res_all['t_pred'].apply(lambda x: get_time_range(x))
    bootstrap = bootstrapping_analysis(res_all)
    if model_name == 'mlp':
        pickle.dump(bootstrap, open(f"../EXP2_{model_name}_{data_name}.pkl", "wb"))
    print("\nData: ", data_name, " Model: ", model_name)
    plot_bootstrapping_results(bootstrap)


if __name__ == '__main__':
    # temporal data
    print("load temporal data... ", end='', flush=True)
    X_t, y_t, l_t, idx_map_t = pickle.load(open("data/tasmc_temporal/tasmc_temporal.pkl", "rb"))
    X_p, y_p, l_p, idx_map_p = pickle.load(
        open("data/proact_temporal/proact_temporal.pkl", "rb"))
    print("loaded.", flush=True)

    # flattened data
    print("load flattened data... ", end="", flush=True)
    X_flat_t = pd.read_csv("data/tasmc_static/tasmc_flat.csv")
    X_flat_p = pd.read_csv("data/proact_static/proact_flat.csv")
    print("loaded.", flush=True)

    m_code = dict(transformer=0, rnn=1, gru=2, mlp=5)
    d_code = dict(proact=0, tasmc=1)

    #
    # run('tasmc', 'rnn')
    # run('tasmc', 'gru')
    # run('tasmc', 'transformer')
    # run('tasmc', 'mlp')
    #
    # run('proact', 'rnn')
    # run('proact', 'gru')
    # run('proact', 'transformer')
    run('proact', 'mlp')
    run('tasmc', 'mlp')
