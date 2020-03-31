import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns

plots_folder = 'ModelComparison/plots/'

res = pd.DataFrame(columns=['iter', 'mae', 'rmse', 'which', 'model'])
for model_name in ['XGB', 'RF', 'LSTM']:
    for number_iter in range(60):
        file = f'C:/Users/Ben/Desktop/Results/train_PROACT_test_TAMC/{model_name}/predictions/{number_iter}.csv'
        cur_res = pd.read_csv(f'C:/Users/Ben/Desktop/Results/train_PROACT_test_TAMC/{model_name}/predictions/{number_iter}.csv')
        for type in ['old', 'new']:
            y_true = cur_res['y_true'].values
            y_preds = cur_res['pred'].values if type == 'old' else cur_res['new_preds'].values
            rmse = mean_squared_error(y_true, y_preds) ** 0.5
            mae = mean_absolute_error(y_true, y_preds)
            res = res.append(dict(iter=number_iter,
                                  mae=mae,
                                  rmse=rmse,
                                  which=type,
                                  model=model_name),
                             ignore_index=True)

res.groupby(['which', 'model']).mean()
res.groupby(['which', 'model']).std()

metric = 'mae'
sns.boxplot(x='model', y=metric, hue='which', data=res.replace(['old', 'new'], ['TAMC', 'PROACT']),
            hue_order=['TAMC', 'PROACT'],
            palette=['lightsteelblue', 'tan'])
legend = plt.legend(title='Train set', loc='upper left', fontsize=10)
legend.get_title().set_fontsize('12')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Model', fontsize=20)
plt.ylabel(metric.upper(), fontsize=20)
plt.savefig(plots_folder + f'{metric}_test_TAMC.png', quality=100, bbox_inches='tight')
plt.show()



for model_name in ['LSTM', 'XGB', 'RF']:
    for metric in ['rmse', 'mae']:
        diff = res.loc[(res['which'] == 'new') & (res['model'] == model_name), metric].values - res.loc[(res['which'] == 'old') & (res['model'] == model_name), metric].values
        plt.plot(diff, lw=0, marker='o', markeredgecolor='k', markersize=9)
        plt.hlines(y=0, xmin=-1, xmax=61, color='red', linestyles='--', lw=3)
        plt.xlim(-0.5, 60.5)
        plt.ylim(-3, 2)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.xlabel('Iteration number', fontsize=20)
        plt.ylabel(f'{metric.upper()} difference', fontsize=20)
        plt.savefig(f'ModelComparison/plots/{metric.upper()}_diff_test_TAMC_{model_name}.png', bbox_inches='tight', quality=100)
        plt.show()

t, p = ttest_rel(res[(res.which == 'old') & (res.model == model_name)][metric], res[(res.which == 'new') & (res.model == model_name)][metric])
# Null: RMSE(old) < RMSE(new)
if (t < 0) and (p/2 < .05):
    print("Reject H0")
print(f't-statistic: {t:.4f}')
print(f'P value: {p:.4f}')


