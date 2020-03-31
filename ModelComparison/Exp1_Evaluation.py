import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.utils import resample


folder = 'C:/Users/Ben/Desktop/Results/'


def get_time_range(t):
    if t <= 6*(365/12):
        return 1
    elif t <= 12*(365/12):
        return 2
    elif t <= 18*(365/12):
        return 3
    else:
        return 4


# Exp1
res = pd.DataFrame()
for database in ['TAMC', 'PROACT']:
    for model_name in ['XGB', 'RF', 'LSTM']:
        for i in range(60):
            try:
                a = pd.read_csv(folder + f'{database}/{model_name}/predictions/{i}.csv')
                a['database'] = database
                a['iter'] = i
                a['model'] = model_name
                a['time_range'] = a['t_pred'].apply(get_time_range)
                a['error'] = (a['pred'] - a['y_true']).abs()
                res = pd.concat([res, a])
            except:
                print(database, model_name, i)

# TAMC
for model in ['RF', 'XGB', 'LSTM']:
    for t in range(1, 5):
        temp = res[(res.database == 'TAMC') & (res.model == model) & (res.time_range == t)]
        rmse = mean_squared_error(temp.y_true, temp.pred) ** 0.5
        print(f'Model: {model}\tTime range: {t}\tRMSE: {rmse:.2f} samples {temp.shape[0]}')

sns.boxplot(x='time_range', y='error', hue='model', data=res)
plt.show()

random.seed(1)
database = 'TAMC'
bootstrap = dict()
for model in ['RF', 'XGB', 'LSTM']:
    bootstrap[model] = dict()
    print(f'::: {model} :::')
    for t in range(1, 6):
        bootstrap[model][t] = dict()
        bootstrap[model][t]['rmse'] = []
        bootstrap[model][t]['mae'] = []
        for i in range(1000):
            if t <= 4:
                samp = resample(res[(res.database == database) & (res.model == model) & (res.time_range == t)], replace=True, n_samples=800)
            else:
                samp = resample(res[(res.database == database) & (res.model == model)], replace=True, n_samples=800)
            rmse = mean_squared_error(samp.y_true, samp.pred) ** 0.5
            mae = mean_absolute_error(samp.y_true, samp.pred)
            bootstrap[model][t]['rmse'].append(rmse)
            bootstrap[model][t]['mae'].append(mae)




for model in ['RF', 'XGB', 'LSTM']:
    print(f":: {model} ::")
    for t in range(1, 6):
        rmse_l = bootstrap[model][t]['rmse']
        mae_l = bootstrap[model][t]['mae']
        print(f'Time range: {t} RMSE: {np.mean(rmse_l):.2f} ({np.std(rmse_l):.2f})\tMAE: {np.mean(mae_l):.2f} ({np.std(mae_l):.2f})')

import pickle
with open('C:/Users/Ben/Desktop/Results/Exp1_TAMC_results.pkl', 'wb') as f:
    pickle.dump([bootstrap, res], f, pickle.HIGHEST_PROTOCOL)



# PROACT

for model in ['RF', 'XGB', 'LSTM']:
    for t in range(1, 5):
        temp = res[(res.database == 'PROACT') & (res.model == model) & (res.time_range == t)]
        if temp.shape[0] == 0:
            rmse=np.nan
        else:
            rmse = mean_squared_error(temp.y_true, temp.pred) ** 0.5
        print(f'Model: {model}\tTime range: {t}\tRMSE: {rmse:.2f} samples {temp.shape[0]}')


import random
from sklearn.utils import resample
random.seed(1)
database = 'PROACT'
bootstrap = dict()
for model in ['RF', 'XGB', 'LSTM']:
    bootstrap[model] = dict()
    print(f'::: {model} :::')
    for t in range(1, 6):
        bootstrap[model][t] = dict()
        bootstrap[model][t]['rmse'] = []
        bootstrap[model][t]['mae'] = []
        for i in range(1000):
            if t <= 4:
                samp = resample(res[(res.database == database) & (res.model == model) & (res.time_range == t)], replace=True, n_samples=800)
            else:
                samp = resample(res[(res.database == database) & (res.model == model)], replace=True, n_samples=800)
            rmse = mean_squared_error(samp.y_true, samp.pred) ** 0.5
            mae = mean_absolute_error(samp.y_true, samp.pred)
            bootstrap[model][t]['rmse'].append(rmse)
            bootstrap[model][t]['mae'].append(mae)

for model in ['RF', 'XGB', 'LSTM']:
    print(f":: {model} ::")
    for t in range(1, 6):
        rmse_l = bootstrap[model][t]['rmse']
        mae_l = bootstrap[model][t]['mae']
        print(f'Time range: {t} RMSE: {np.mean(rmse_l):.2f} ({np.std(rmse_l):.2f})\tMAE: {np.mean(mae_l):.2f} ({np.std(mae_l):.2f})')

import pickle
with open('C:/Users/Ben/Desktop/Results/Exp1_PROACT_results.pkl', 'wb') as f:
    pickle.dump([bootstrap, res], f, pickle.HIGHEST_PROTOCOL)

