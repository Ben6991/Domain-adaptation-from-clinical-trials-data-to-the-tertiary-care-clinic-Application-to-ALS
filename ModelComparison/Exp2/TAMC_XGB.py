import sys
import os

sys.path.append('/gpfs0/boaz/users/benhada/home/Master/')
sys.path.append('C:/Users/benhada/master/Models/')
sys.path.append('/gpfs0/boaz/users/benhada/home/Master/Models')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor
from xgboost.callback import early_stop

model_name = 'XGB'

number_iter = 0
folder = 'C:/Users/Ben/Desktop/Results/'
tuning_path = folder + f'TAMC/{model_name}/tuning_files/tuning_0.pkl'
new_preds_path = folder + f'Exp2/{model_name}/predictions/'
train_test_file = folder + 'TAMC_2_train_test_split.pkl'
TAMC_path = folder + 'TAMC2.csv'
TAMC = pd.read_csv(TAMC_path)

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features

# ----------------------------------------- #
# --------------- train-val --------------- #
# ----------------------------------------- #

with open(train_test_file, 'rb') as f:
    split = pickle.load(f)
train_pats = split[number_iter]['train_pats']
test_pats = split[number_iter]['test_pats']

# Split Train-Validation-Test
n_val_samples = int(0.1*len(train_pats))
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=32)

# create dicitonaries
val_dict = ts.create_mapping(TAMC, val_pats, 2, 10, 32, 'n_samples')
train_dict = ts.enrichment_by_ID_list(TAMC, train_pats, 2, random_state=1)

# create Train-Validation for early stopping
exlude_features = ['aos', 'sos', 'aoc', 'kurtosis', 'skewness']
print(f'Create Xy train and validation', flush=True)
train_data = ts.temporal_to_static(TAMC, train_dict, temporal_features, constant_features, exlude_features=exlude_features)
validation_data = ts.temporal_to_static(TAMC, val_dict, temporal_features, constant_features, exlude_features=exlude_features)

# ----------------------------------------------- #
# ---------------- Random search ---------------- #
# ----------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_val = validation_data.drop(['ID', 'y'], 1).values
y_val = validation_data['y'].values

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ---------------------------- #
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(tuning_path, 'rb') as f:
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
                     n_estimators=5000,
                     objective='reg:squarederror')

# Fit model using early stopping
early = early_stop(stopping_rounds=30, maximize=False)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          callbacks=[early])

# ------------------------------------------------------------------- #
# ------------------------- Test prediction ------------------------- #
# ------------------------------------------------------------------- #

print("Evaluation", flush=True)
test_dict = ts.create_mapping(TAMC, test_pats, 2, np.inf, 32, 'n_samples')
test_data = ts.temporal_to_static(TAMC, test_dict, temporal_features, constant_features, exlude_features=exlude_features)

X_test = test_data.drop(['ID', 'y'], 1).values
y_test = test_data['y'].values

preds = model.predict(X_test, ntree_limit=model.get_booster().best_iteration)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID_i in test_data.ID.unique():
    ID, idx = ID_i.split('_')
    x_v, y_v = test_dict[ID][int(idx)]
    summary = summary.append(dict(ID=ID,
                                  xlast=x_v[-1],
                                  t_pred=ts.time_between_records(TAMC, x_v[-1], y_v[-1], ID),
                                  yvis=y_v[-1],
                                  y_true=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == y_v[0]), 'y'].item(),
                                  y_last=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == x_v[-1]), 'y'].item()),
                             ignore_index=True)
summary['pred'] = preds

summary.to_csv(new_preds_path + f'{number_iter}.csv', index=False)
import sys
import os

sys.path.append('/gpfs0/boaz/users/benhada/home/Master/')
sys.path.append('C:/Users/benhada/master/Models/')
sys.path.append('/gpfs0/boaz/users/benhada/home/Master/Models')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor
from xgboost.callback import early_stop

model_name = 'XGB'

number_iter = 0
folder = 'C:/Users/Ben/Desktop/Results/'
tuning_path = folder + f'TAMC/{model_name}/tuning_files/tuning_0.pkl'
new_preds_path = folder + f'Exp2/{model_name}/predictions/'
train_test_file = folder + 'TAMC_2_train_test_split.pkl'
TAMC_path = folder + 'TAMC2.csv'
TAMC = pd.read_csv(TAMC_path)

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features

# ----------------------------------------- #
# --------------- train-val --------------- #
# ----------------------------------------- #

with open(train_test_file, 'rb') as f:
    split = pickle.load(f)
train_pats = split[number_iter]['train_pats']
test_pats = split[number_iter]['test_pats']

# Split Train-Validation-Test
n_val_samples = int(0.1*len(train_pats))
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=32)

# create dicitonaries
val_dict = ts.create_mapping(TAMC, val_pats, 2, 10, 32, 'n_samples')
train_dict = ts.enrichment_by_ID_list(TAMC, train_pats, 2, random_state=1)

# create Train-Validation for early stopping
exlude_features = ['aos', 'sos', 'aoc', 'kurtosis', 'skewness']
print(f'Create Xy train and validation', flush=True)
train_data = ts.temporal_to_static(TAMC, train_dict, temporal_features, constant_features, exlude_features=exlude_features)
validation_data = ts.temporal_to_static(TAMC, val_dict, temporal_features, constant_features, exlude_features=exlude_features)

# ----------------------------------------------- #
# ---------------- Random search ---------------- #
# ----------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_val = validation_data.drop(['ID', 'y'], 1).values
y_val = validation_data['y'].values

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ---------------------------- #
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(tuning_path, 'rb') as f:
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
                     n_estimators=5000,
                     objective='reg:squarederror')

# Fit model using early stopping
early = early_stop(stopping_rounds=30, maximize=False)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          callbacks=[early])

# ------------------------------------------------------------------- #
# ------------------------- Test prediction ------------------------- #
# ------------------------------------------------------------------- #

print("Evaluation", flush=True)
test_dict = ts.create_mapping(TAMC, test_pats, 2, np.inf, 32, 'n_samples')
test_data = ts.temporal_to_static(TAMC, test_dict, temporal_features, constant_features, exlude_features=exlude_features)

X_test = test_data.drop(['ID', 'y'], 1).values
y_test = test_data['y'].values

preds = model.predict(X_test, ntree_limit=model.get_booster().best_iteration)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID_i in test_data.ID.unique():
    ID, idx = ID_i.split('_')
    x_v, y_v = test_dict[ID][int(idx)]
    summary = summary.append(dict(ID=ID,
                                  xlast=x_v[-1],
                                  t_pred=ts.time_between_records(TAMC, x_v[-1], y_v[-1], ID),
                                  yvis=y_v[-1],
                                  y_true=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == y_v[0]), 'y'].item(),
                                  y_last=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == x_v[-1]), 'y'].item()),
                             ignore_index=True)
summary['pred'] = preds

summary.to_csv(new_preds_path + f'{number_iter}.csv', index=False)