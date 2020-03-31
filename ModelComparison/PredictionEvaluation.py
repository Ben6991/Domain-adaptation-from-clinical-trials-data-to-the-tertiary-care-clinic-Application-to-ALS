import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import pickle

res_dict = dict()
res = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_true', 't_pred', 'pred', 'database', 'model_name'])
for database in ['PROACT', 'TAMC']:
    res_dict[database] = dict()
    for model_name in ['XGB', 'RF', 'LSTM']:
        res_dict[database][model_name] = dict()
        for iter in range(60):
            try:
                if database == 'TAMC':
                    cur_res = pd.read_csv(f'C:/Users/Ben/Desktop/Results/new_{database}/{model_name}/predictions/{iter}.csv')
                else:
                    cur_res = pd.read_csv(f'C:/Users/Ben/Desktop/Results/{database}/{model_name}/predictions/{iter}.csv')
                res_dict[database][model_name][iter] = cur_res
                # cur_res['database'] = database
                # cur_res['model'] = moodel_name
                # a = res.append(cur_res)
                # res = pd.concat([res, a])
                # print(res.shape[0])
            except:
                print(f"\tNot exist\ndatabase: {database}\nmodel: {model_name}\niter: {iter}")

# -------------------- Collect results --------------------

metrics = pd.DataFrame(columns=['database', 'model_name', 'iter', 'rmse'])
for database in ['TAMC', 'PROACT']:
    for model_name in ['LSTM', 'XGB', 'RF']:
        for iter in [*res_dict[database][model_name].keys()]:
            temp = res_dict[database][model_name][iter]
            rmse = mean_squared_error(temp['pred'], temp['y_true']) ** 0.5
            mae = mean_absolute_error(temp['pred'], temp['y_true'])
            pcc = r2_score(temp['pred'], temp['y_true'])
            metrics = metrics.append(dict(database=database,
                                          model_name=model_name,
                                          iter=iter,
                                          rmse=rmse,
                                          mae=mae,
                                          pcc=pcc),
                                     ignore_index=True)

metric = 'rmse'
print(f'Database\t\tModel\t\t{metric}')
for database in ['TAMC', 'PROACT']:
    for model_name in ['XGB', 'RF', 'LSTM']:
        mean_ = metrics[(metrics.database == database) & (metrics.model_name == model_name)][metric].mean()
        std_ = metrics[(metrics.database == database) & (metrics.model_name == model_name)][metric].std()
        print(f'{database}\t\t{model_name}\t\t\t{mean_:.2f} ({std_:.2f})')

# Model comparison
metric = 'mae'  # 'rmse', 'mae' or 'pcc'
database = 'TAMC'  # 'PROACT' or 'TAMC
fig = plt.figure(figsize=(5, 3))
sns.boxplot(x='model_name', y=metric, data=metrics[metrics.database == database],
            hue_order=['LSTM', 'XGB', 'RF'],
            palette=['C0', 'C1', 'C2'])
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.xlabel('Model', fontsize=15)
plt.ylabel(metric.upper(), fontsize=15)
# plt.savefig(f'ModelComparison/plots/comparison_{database}_{metric}.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# time_range:
#   0:[0, 6)      1:[6,12)  2: [12, 18) 3: [18,inf)
#

def create_time_range(t, days=True):
    days_in_months = 365/12
    if t < 6*days_in_months:
        return 0
    elif t < 12*days_in_months:
        return 1
    elif t < 18*days_in_months:
        return 2
    else:
        return 3


for database in ['PROACT', 'TAMC']:
    for model_name in ['XGB', 'LSTM', 'RF']:
        for iter in [*res_dict[database][model_name].keys()]:
            res_dict[database][model_name][iter]['time_range'] = res_dict[database][model_name][iter]['t_pred'].apply(create_time_range)
            res_dict[database][model_name][iter]['error'] = (res_dict[database][model_name][iter]['pred'] - res_dict[database][model_name][iter]['y_true']).abs()


yy = pd.DataFrame(columns=['database', 'model_name', 'iter', 'time_range', 'rmse'])
for database in ['TAMC', 'PROACT']:
    for model_name in ['XGB', 'RF', 'LSTM']:
        for iter in [*res_dict[database][model_name].keys()]:
            for t_r in range(4):
                temp_data = res_dict[database][model_name][iter]
                temp_data = temp_data[temp_data['time_range'] == t_r]
                if temp_data.shape[0] == 0:
                    rmse = np.nan
                    mae = np.nan
                else:
                    rmse = mean_squared_error(temp_data['pred'], temp_data['y_true']) ** 0.5
                    mae = mean_absolute_error(temp_data['pred'], temp_data['y_true'])
                yy = yy.append(dict(database=database,
                                    model_name=model_name,
                                    iter=iter,
                                    time_range=t_r,
                                    rmse=rmse,
                                    mae=mae),
                               ignore_index=True)

database = 'TAMC'
metric='mae'
sns.boxplot(x='time_range', y=metric, hue='model_name', data=yy[(yy.database==database)],
            hue_order=['LSTM', 'XGB', 'RF'],
            palette=['C0', 'C1', 'C2'])
plt.xticks([0, 1, 2, 3], ['[0, 6)', '[6, 12)', '[12, 18)', '>18'], fontsize=15)
plt.xlabel('Prediction time (months)', fontsize=15)
plt.legend(title='Model')
plt.ylabel(metric.upper(), fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(f'ModelComparison/plots/{database}_{metric}_time_range.png', bbox_inches='tight')
plt.show()
#

database = 'TAMC'; model_name = 'XGB'
a = pd.concat([res_dict[database][model_name][i] for i in range(60)])
a.drop_duplicates(inplace=True)
plt.title(f'{database} {model_name}')
sns.lineplot(x='xlast', y='error', hue='time_range', data=a, estimator='mean', ci=50)
plt.show()


# # polynim cuuve
# for time_range in [0, 1, 2, 3]:
#     z = np.polyfit(a[a['time_range'] == time_range]['xlast'], a[a['time_range'] == time_range]['error'], deg=3)
#     max_x = a[a['time_range'] == time_range]['xlast'].max()
#     p = np.poly1d(z)
#     plt.plot(range(0, max_x), p(range(0, max_x)))
# plt.legend(['0', '1', '2', '3'])
# plt.show()
#
