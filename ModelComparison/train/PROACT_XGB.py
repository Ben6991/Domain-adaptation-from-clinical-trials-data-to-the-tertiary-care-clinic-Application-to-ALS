import sys
import os

sys.path.append('/gpfs0/boaz/users/benhada/home/Master/')
sys.path.append('/gpfs0/boaz/users/benhada/home/Master/Models')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor
from xgboost.callback import early_stop

database_name = 'PROACT'
model_name = 'XGB'

# ----------------------------------------- #
# ------------ configurations ------------ #
# ----------------------------------------- #

number_iter = 0
folder = 'C:/Users/benhada/Desktop/Results/'
data_file = folder + 'TAMC1.csv'
TUNING_PATH = folder + f'{database_name}/{model_name}/tuning_files/tuning_{number_iter}.pkl'

# ----------------------------------------- #
# --------------- Load data --------------- #
# ----------------------------------------- #

data = pd.read_csv(data_file)

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

temporal_features = ['ALSFRS_Total', 'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing',
                     'Q4_Handwriting', 'Q5_Cutting', 'Q6_Dressing_and_Hygiene',
                     'Q7_Turning_in_Bed', 'Q8_Walking', 'Q9_Climbing_Stairs',
                     'Q10_Respiratory', 'FVC', 'CK', 'Creatinine', 'Phosphorus',
                     'Alkaline.Phosphatase', 'Albumin', 'Bilirubin', 'Chloride',
                     'Hematocrit', 'Hemoglobin', 'Potassium', 'Protein', 'Glucose',
                     'Calcium', 'Sodium', 'Platelets']

constant_features = ['Age', 'Onset.time', 'Onset.Bulbar', 'Onset.Limb', 'Sex.Male']

features = temporal_features + constant_features

# Drop all missing values
ts.drop_by_number_of_records(data, 3, inplace=True)
ts.add_ordering(data, inplace=True)

data = data[['ID', 'time', 'no'] + features + ['y']]
data.reset_index(drop=True, inplace=True)

# ------------------------------------------ #
# ------------ Split train-test ------------ #
# ------------------------------------------ #

print(f'>> Split Train-Validation-Test', flush=True)
n_test_samples = int(0.15*data['ID'].nunique())
n_val_samples = int(0.15*data['ID'].nunique())
n_early_stopping_samples = int(0.1*data['ID'].nunique())
train_pats, test_pats = train_test_split(data['ID'].unique(), test_size=n_test_samples, random_state=number_iter)
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=number_iter)
train_pats, early_stop_pats = train_test_split(train_pats, test_size=n_early_stopping_samples, random_state=number_iter)

# create test dictionary (unique for each iteration)
print(f'>> Split x,y visits for each validation and test patient', flush=True)
test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random')
val_dict = ts.create_mapping(data, val_pats, 2, 4, 32, 'r_random')
early_stopping_dict = ts.create_mapping(data, early_stop_pats, 2, 4, 32, 'r_random')

# ------------------------------------------------------------------------------------ #
# ------------------------------ Hyper-parameter tuning ------------------------------ #
# ------------------------------------------------------------------------------------ #

# enrichment
print(f'>> Training data enrichment', flush=True)
train_dict = ts.enrichment_by_ID_list(data,
                                      list_IDs=train_pats,
                                      min_x_visits=2,
                                      progress_bar=True)

# create Train-Validation for early stopping
print(f'Create Xy train', flush=True)
train_data = ts.temporal_to_static(data, train_dict, temporal_features, constant_features)
print(f'Create Xy for early stopping', flush=True)
early_stopping_data = ts.temporal_to_static(data, early_stopping_dict, temporal_features, constant_features)
print(f'Create Xy validation', flush=True)
validation_data = ts.temporal_to_static(data, val_dict, temporal_features, constant_features)
# print(f'Create Xy test', flush=True)
# test_data = ts.temporal_to_static(data, test_dict, temporal_features, constant_features)

# ----------------------------------------------- #
# ---------------- Random search ---------------- #
# ----------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_early = early_stopping_data.drop(['ID', 'y'], 1).values
y_early = early_stopping_data['y'].values
X_val = validation_data.drop(['ID', 'y'], 1).values
y_val = validation_data['y'].values

param_dist = {'max_depth': np.arange(2, 30),
              'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
              'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
              'gamma': [0, 0.25, 0.5, 1.0],
              'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}


n_iter = 60
param_sampler = ParameterSampler(param_dist, n_iter, random_state=123)
min_validation_mse = np.inf
tuning_table = pd.DataFrame(columns=[*param_dist.keys()] + ['validation_mse'])

for i, params in enumerate(param_sampler):
    try:
        print(f"----------------------------------------------------------- ")
        print(f"------ Random search iteration number {i+1}/{n_iter} ------ ")
        print(f"----------------------------------------------------------- ")
        ts.print_configuration(params)

        # Create model
        model = XGBRegressor(max_depth=params['max_depth'],
                             learning_rate=params['learning_rate'],
                             subsample=params['subsample'],
                             colsample_bytree=params['colsample_bytree'],
                             colsample_bylevel=params['colsample_bylevel'],
                             min_child_weight=params['min_child_weight'],
                             gamma=params['gamma'],
                             reg_lambda=params['reg_lambda'],
                             n_estimators=3000,
                             objective='reg:squarederror')

        # Fit model using early stopping
        early = early_stop(stopping_rounds=30, maximize=False)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_early, y_early)],
                  callbacks=[early])

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

    except:
        pass

# Save tuning table
with open(TUNING_PATH, 'wb') as f:
    pickle.dump(tuning_table, f, pickle.HIGHEST_PROTOCOL)


