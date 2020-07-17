"""
1.  process all the tuning files and create "best_configs.pkl" with the best configuration
    for each model and for each database.

2.  hyper-parameter analysis - which hyper-parameters are the most important to tune.
"""

import os
import pickle

import pandas as pd


def modify_res(results):
    if modelnm == 'rnn':
        results['norm_type'] = results['norm_type'].apply(lambda x: 1 if isinstance(x, str) else 0)


def get_hparams(m: str):
    if m == 'rnn' or m == 'gru':
        return ['lr', 'hidden_size', 'batch_size', 'num_layers', 'learn_initial_state', 'weight_decay', 'norm_type']
    if m == 'mlp':
        return ['n_hidden_layers', 'n_hidden_units', 'activation', 'batch_size', 'lr', 'regularizer_val', 'dropout',
                'batch_norm']
    if m == 'transformer':
        return ['lr', 'batch_size', 'num_layers', 'nhead', 'dim_feedforward']


def initialization():
    best_configs = dict()
    for data in ['tasmc', 'proact']:
        best_configs[data] = dict()
    return best_configs


if __name__ == '__main__':

    best_configs = initialization()

    a = dict(proact=0, tasmc=1)
    b = dict(transformer=0, rnn=1, gru=2, mlp=5)

    for modelnm in ['rnn', 'gru', 'mlp', 'transformer']:
        for datanm in ['tasmc', 'proact']:
            best_configs[datanm][modelnm] = dict()
            hparams = get_hparams(modelnm)
            res = pd.DataFrame()
            for i in range(60):
                path = f'results/{a[datanm]}{b[modelnm]}0/tuning_{i}.csv'
                if not os.path.exists(path):
                    path = f'results/{a[datanm]}{b[modelnm]}0/{i}.csv'
                    print("ok")
                if os.path.exists(path):
                    res_i = pd.read_csv(path)
                    hparams_values = res_i.sort_values('validation_mse', ascending=True).head(1)[
                        hparams].values.ravel().tolist()
                    best_config = dict(zip(hparams, hparams_values))
                    best_configs[datanm][modelnm][i] = best_config
                else:
                    print(f'{path} doesnt exist')

    best_configs_file = "f18_best_configs.pkl"
    print(f"savling {best_configs_file}")
    pickle.dump(best_configs, open(best_configs_file, "wb"))
