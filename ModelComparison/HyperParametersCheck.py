import pandas as pd
import pandas as pd
import pickle

tune = pd.DataFrame()
for i in range(60):
    with open(f'C:/Users/Ben/Desktop/Results/old_TAMC/XGB/tuning_files/tuning_{i}.pkl', 'rb') as f:
        a = pickle.load(f)
        a['iter'] = i
        a['No'] = range(a.shape[0])
        tune = pd.concat([tune, a])

sorted_mean = tune[['No', 'validation_mse']].groupby('No').mean().sort_values('validation_mse').reset_index()
sorted_min = tune[['No', 'validation_mse']].groupby('No').min().sort_values('validation_mse').reset_index()
sorted_max = tune[['No', 'validation_mse']].groupby('No').max().sort_values('validation_mse').reset_index()


import numpy as np
counter = np.zeros(60)
for i in range(60):
    number = tune[tune['iter'] == i].sort_values('validation_mse').head(1).No.item()
    counter[number] += 1


sorted_mean.head(20)