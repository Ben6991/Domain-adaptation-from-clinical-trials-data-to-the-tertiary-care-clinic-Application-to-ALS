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

number_iter = 0
folder = f'C:/Users/benhada/Desktop/Results/'
data_file = f'C:/Users/benhada/Desktop/Data/PROACT/data.csv'
MODEL_PATH = f'C:/Users/benhada/Desktop/Results/{database_name}/{model_name}/models/model_{number_iter}.model'
TUNING_PATH = f'C:/Users/benhada/Desktop/Results/{database_name}/{model_name}/tuning_files/tuning_{number_iter}.pkl'
summary_csv = f'C:/Users/benhada/Desktop/Results/{database_name}/{model_name}/predictions/{number_iter}.csv'

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

# create test dictionary (unique for each iteration)
print(f'>> Split x,y visits for each validation and test patient', flush=True)
test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random')
val_dict = ts.create_mapping(data, val_pats, 2, 4, 32, 'r_random')

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
print(f'Create Xy validation', flush=True)
validation_data = ts.temporal_to_static(data, val_dict, temporal_features, constant_features)
print(f'Create Xy test', flush=True)
test_data = ts.temporal_to_static(data, test_dict, temporal_features, constant_features)

# ----------------------------------------------- #
# ---------------- Random search ---------------- #
# ----------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_val = validation_data.drop(['ID', 'y'], 1).values
y_val = validation_data['y'].values
X_test = test_data.drop(['ID', 'y'], 1).values
y_test = test_data['y'].values

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ----------------------------#
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(TUNING_PATH, 'rb') as f:
    a = pickle.load(f)
a = a[a.validation_mse == a.validation_mse.min()]
params = dict(zip(a.columns.tolist(), a.values[0].tolist()))

# Create model
model = XGBRegressor(max_depth=int(params['max_depth']),
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
          eval_set=[(X_train, y_train), (X_val, y_val)],
          callbacks=[early])

# Save model
model.save_model(MODEL_PATH)

# Test prediction
predictions = model.predict(X_test)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID_i in test_data.ID.unique():
    ID, idx = ID_i.split('_')
    x_v, y_v = test_dict[int(ID)][int(idx)]
    summary = summary.append(dict(ID=ID,
                                  xlast=x_v[-1],
                                  t_pred=ts.time_between_records(data, x_v[-1], y_v[-1], int(ID)),
                                  yvis=y_v[-1],
                                  y_true=data.loc[(data['ID'] == int(ID)) & (data['no'] == y_v[0]), 'y'].item(),
                                  y_last=data.loc[(data['ID'] == int(ID)) & (data['no'] == x_v[-1]), 'y'].item()),
                             ignore_index=True)

summary['pred'] = predictions

summary.to_csv(summary_csv, index=False)

