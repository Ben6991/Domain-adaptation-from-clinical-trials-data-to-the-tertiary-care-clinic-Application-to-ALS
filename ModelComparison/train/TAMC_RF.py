import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestRegressor

number_iter = 0
database = 'TAMC'
model_name = 'RF'

folder = 'C:/Users/benhada/Desktop/Results/'
data_file = folder + f'{database}/Data/TAMC.csv'
MODEL_PATH = folder + f'{database}/{model_name}/models/{number_iter}.pkl'
summary_csv = folder + f'{database}/{model_name}/predictions/{number_iter}.csv'
tuning_file = folder + f'{database}/{model_name}/tuning_files/tuning_{number_iter}.pkl'
importance_csv = folder + f'{database}/{model_name}/feature_importances/{number_iter}.csv'
train_test_split_info = folder + 'TAMC_1_train_test_split.pkl'
data = pd.read_csv(data_file)

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

temporal_features = ['RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
                     'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt', 'LFingerAbduct',
                     'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi', 'RAnklePlantar',
                     'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar', 'Speech',
                     'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Turning',
                     'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']

constant_features = ['AgeOnset', 'Gender', 'GenderNA', 'Sport', 'SportNA',
                     'OnsetSite', 'OnsetSiteNA', 'Familial', 'FamilialNA',
                     'Dementia', 'DementiaNA', 'TimeSinceOnset', 'Age',
                     'isSmoking', 'isSmokingNA']

features = temporal_features + constant_features

data = data[['ID', 'time', 'no'] + features + ['y']]

# ------------------------------------------------ #
# ------------ Split train-validation ------------ #
# ------------------------------------------------ #

with open(train_test_split_info, 'rb') as f:
    split = pickle.load(f)
train_pats = split[number_iter]['train_pats']

train_pats, val_pats = train_test_split(train_pats, test_size=0.1, random_state=32)
train_pats, early_pats = train_test_split(train_pats, test_size=0.1, random_state=32)

# create test dictionary (unique for each iteration)
print(f'Create dictionaries', flush=True)
val_dict = ts.create_mapping(data, val_pats, 2, 4, 32, 'r_random')
early_stopping_dict = ts.create_mapping(data, early_pats, 2, 4, 32, 'r_random')
train_dict = ts.enrichment_by_ID_list(data, train_pats, 2, random_state=1)

# ------------------------------------------------------------------------------------ #
# ------------------------------ Hyper-parameter tuning ------------------------------ #
# ------------------------------------------------------------------------------------ #

print(f'Create Xy train', flush=True)
exlude_features = ['skewness', 'kurtosis', 'aos', 'soc', 'sos']
train_data = ts.temporal_to_static(data, train_dict, temporal_features, constant_features, exlude_features=exlude_features)
print(f'Create Xy for early stopping', flush=True)
early_stopping_data = ts.temporal_to_static(data, early_stopping_dict, temporal_features, constant_features, exlude_features=exlude_features)
print(f'Create Xy validation', flush=True)
validation_data = ts.temporal_to_static(data, val_dict, temporal_features, constant_features, exlude_features=exlude_features)

# ----------------------------------------------- #
# ---------------- Random search ---------------- #
# ----------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_early = early_stopping_data.drop(['ID', 'y'], 1).values
y_early = early_stopping_data['y'].values
X_val = validation_data.drop(['ID', 'y'], 1).values
y_val = validation_data['y'].values


param_dist = {'bootstrap': [True, False],
              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10],
              'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

n_iter = 60
param_sampler = ParameterSampler(param_dist, n_iter, random_state=123)
min_validation_mse = np.inf
tuning_table = pd.DataFrame(columns=[*param_dist.keys()] + ['validation_mse'])

for i, params in enumerate(param_sampler):
    print(f"----------------------------------------------------------- ", flush=True)
    print(f"------ Random search iteration number {i+1}/{n_iter} ------ ", flush=True)
    print(f"----------------------------------------------------------- ", flush=True)
    print(params)

    # Create model
    model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                  max_depth=params['max_depth'],
                                  max_features=params['max_features'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  min_samples_split=params['min_samples_split'],
                                  bootstrap=params['bootstrap'])

    # Fit model using early stopping
    model.fit(X_train, y_train)

    # Validation evaluation
    y_val_preds = model.predict(X_val)
    cur_mse = mean_squared_error(y_val_preds, y_val)

    # save best score
    if cur_mse < min_validation_mse:
        min_validation_mse = cur_mse
        print(f"Validation decreased to {min_validation_mse}", flush=True)

    # save configuration
    params['validation_mse'] = cur_mse
    tuning_table = tuning_table.append(params, ignore_index=True)

# Save tuning table
with open(TUNING_PATH, 'wb') as f:
    pickle.dump(tuning_table, f, pickle.HIGHEST_PROTOCOL)

print("Finished successfully")
