import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.utils import resample

folder = 'C:/Users/Ben/Desktop/Results/Exp2/'


def get_time_range(t):
    if t <= 6*(365/12):
        return 1
    elif t <= 12*(365/12):
        return 2
    elif t <= 18*(365/12):
        return 3
    else:
        return 4


res = pd.DataFrame()
for train_set in ['TAMC', 'PROACT']:
    for model_name in ['XGB', 'RF', 'LSTM']:
        for i in range(60):
            try:
                a = pd.read_csv(folder + f'{train_set}_{model_name}/predictions/{i}.csv')
                a['train_set'] = train_set
                a['iter'] = i
                a['model'] = model_name
                a['time_range'] = a['t_pred'].apply(get_time_range)
                a['error'] = (a['pred'] - a['y_true']).abs()
                res = pd.concat([res, a])
            except:
                print(train_set, model_name, i)


###########
random.seed(1)
bootstrap = dict()
for train_set in ['TAMC', 'PROACT']:
    bootstrap[train_set] = dict()
    for model in ['RF', 'XGB', 'LSTM']:
        bootstrap[train_set][model] = dict()
        print(f'::: {model} :::')
        for t in range(1, 6):
            bootstrap[train_set][model][t] = dict()
            bootstrap[train_set][model][t]['rmse'] = []
            bootstrap[train_set][model][t]['mae'] = []
            for i in range(100):
                if t <= 4:
                    samp = resample(res[(res.train_set == train_set) & (res.model == model) & (res.time_range == t)], replace=True, n_samples=800)
                else:
                    samp = resample(res[(res.train_set == train_set) & (res.model == model)], replace=True, n_samples=800)
                rmse = mean_squared_error(samp.y_true, samp.pred) ** 0.5
                mae = mean_absolute_error(samp.y_true, samp.pred)
                bootstrap[train_set][model][t]['rmse'].append(rmse)
                bootstrap[train_set][model][t]['mae'].append(mae)
###########



res_to_plot = pd.DataFrame()
for train_set in ['PROACT', 'TAMC']:
    for model in ['XGB', 'RF', 'LSTM']:
        for t in range(1, 6):
            for i in range(len(bootstrap[train_set][model][t]['rmse'])):
                res_to_plot = res_to_plot.append(dict(train_set=train_set,
                                                      model=model,
                                                      time_range=t,
                                                      rmse=bootstrap[train_set][model][t]['rmse'][i],
                                                      mae=bootstrap[train_set][model][t]['mae'][i]),
                                                 ignore_index=True)


res_to_plot['train_set'].replace('TAMC', 'TASMC', inplace=True)
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 3.5), sharey=True, sharex=True)
for i, model in enumerate(['XGB', 'RF', 'LSTM']):
    pt = sns.boxplot(x='time_range', y='rmse', hue='train_set',
                data=res_to_plot[(res_to_plot.time_range <= 4) & (res_to_plot.model == model)],
                ax=axes[i])
    axes[i].set_title(model, fontsize=16)
    if i == 0:
        axes[i].set_ylabel("RMSE", fontsize=12)
    else:
        axes[i].set_ylabel("")
    axes[i].set_xlabel("")
    axes[i].set_xticks([0, 1, 2, 3], ['(0, 6]', '(6, 12]', '(12, 18]', '(18, 24]'])
    axes[i].set_xticklabels(['(0, 6]', '(6, 12]', '(12, 18]', '(18, 24]'])
    if i == 0:
        legend = axes[i].legend(title='Train set', loc='upper left', fontsize=10)
        legend.get_title().set_fontsize('11')
    else:
        pt.legend_.remove()
    axes[i].grid(axis='y', alpha=0.5)

fig.text(0.5, -0.02, 'Time range (months)', fontsize=15, ha='center')
plt.tight_layout()
plt.savefig(f'ModelComparison/plots/XGB_exp2.png', bbox_inches='tight')
plt.show()


for train_set in ['TAMC', 'PROACT']:
    for model in ['RF', 'XGB', 'LSTM']:
        print(f":: {model} {train_set}::")
        for t in range(1, 6):
            rmse_l = bootstrap[train_set][model][t]['rmse']
            mae_l = bootstrap[train_set][model][t]['mae']
            print(f'Time range: {t} RMSE: {np.mean(rmse_l):.2f} ({np.std(rmse_l):.2f})\tMAE: {np.mean(mae_l):.2f} ({np.std(mae_l):.2f})')



import pickle
with open('C:/Users/Ben/Desktop/Results/Exp2_results.pkl', 'wb') as f:
    pickle.dump([bootstrap, res], f, pickle.HIGHEST_PROTOCOL)




