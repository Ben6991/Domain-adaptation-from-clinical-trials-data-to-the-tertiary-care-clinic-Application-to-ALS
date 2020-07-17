import pickle
import random

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample


def get_time_range(t):
    period = 4
    if t <= 6 * (365 / 12):
        period = 1
    elif t <= 12 * (365 / 12):
        period = 2
    elif t <= 18 * (365 / 12):
        period = 3
    return period


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


def collect_results():
    res = pd.DataFrame()
    for i in range(60):
        cur = pd.read_csv(folder + f'{i}.csv')
        res = pd.concat([res, cur])
    res['t_period'] = res['t_pred'].apply(get_time_range)
    res.rename(columns=dict(y_true='y'), inplace=True)
    boot = bootstrapping_analysis(res)
    return boot


res = pd.DataFrame()

for data in ['tasmc', 'proact']:
    for model in ['xgb', 'lstm', 'rf']:
        folder = f'exp2_old/{model}_only_{data}/'
        boot = collect_results()
        ts = [*boot]
        ts.sort()
        res = pd.concat([res, pd.concat(
            [pd.DataFrame(dict(model=model, data=data.upper(), rmse=boot[t]['rmse'], t=t)) for t in [1, 2, 3, 4]])])

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x='t', y='rmse', hue='data', data=res[res['model'] == 'rf'])
plt.show()

pro = pickle.load(open("EXP2_mlp_proact.pkl", "rb"))
ta = pickle.load(open("EXP2_mlp_tasmc.pkl", "rb"))

aa = pd.concat([pd.DataFrame(dict(model='mlp', data='PROACT', t=t, rmse=pro[t]['rmse'])) for t in [1, 2, 3, 4]])
res = pd.concat([res, aa])
bb = pd.concat([pd.DataFrame(dict(model='mlp', data='TASMC', t=t, rmse=ta[t]['rmse'])) for t in [1, 2, 3, 4]])
res = pd.concat([res, bb])


titles = ['XGB', 'RF', 'MLP', 'LSTM']
fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(26, 4))
plt.subplots_adjust(wspace=0.09)
for i, model in enumerate(['xgb', 'rf', 'mlp', 'lstm']):
    sns.boxplot(x='t', y='rmse', hue='data', data=res[res['model'] == model], ax=axes[i], hue_order=['PROACT', 'TASMC'])
    axes[i].set_title(titles[i], fontsize=15)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].grid(axis='y', alpha=0.1)
    axes[i].set_xticklabels([1, 2, 3, 4], fontsize=18)
    axes[i].set_yticklabels([2, 3, 4, 5, 6, 7, 8], fontsize=18)
plt.annotate(xy=(0.5, 0.5), s='asdasd')
axes[0].set_ylabel('RMSE', fontsize=20)

fig.text(0.5, -0.01, 'Time range', va='center', ha='center', fontsize=20)

plt.savefig('C:/Users/Ben/Desktop/BOX.png', quality=100, bbox_inches='tight')
plt.clf()
