import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
import pickle
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from xgboost.callback import early_stop

number_iter = 0
data_name = 'TAMC'
model_name = 'XGB'

folder = 'C:/Users/benhada/Desktop/Results/'

data_file = folder + f'TAMC1.csv'
MODEL_PATH = folder + f'{data_name}/{model_name}/models/{number_iter}.model'
summary_csv = folder + f'{data_name}/{model_name}/predictions/{number_iter}.csv'
tuning_file = folder + f'{data_name}/{model_name}/tuning_files/tuning_{number_iter}.pkl'
importance_csv = folder + f'{data_name}/{model_name}/feature_importances/{number_iter}.csv'
train_test_split_info = folder + 'TAMC_1_train_test_split.pkl'
data = pd.read_csv(data_file)

# ----------------------------------------- #
# ---------- Get Best parameters ---------- #
# ----------------------------------------- #

with open(tuning_file, 'rb') as f:
    tuning_table = pickle.load(f)
values = tuning_table.loc[np.argmin(tuning_table['validation_mse'])].values.tolist()
parameters = tuning_table.columns.tolist()
params = dict(zip(parameters, values))

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

# ----------------------------------------- #
# --------- Train-Validation-Test --------- #
# ----------------------------------------- #

with open(train_test_split_info, 'rb') as f:
    split = pickle.load(f)
train_pats = split[number_iter]['train_pats']
test_pats = split[number_iter]['test_pats']

train_pats, early_pats = train_test_split(train_pats, test_size=0.1, random_state=32)

# create test dictionary (unique for each iteration)
print(f'Create dictionaries', flush=True)
early_stopping_dict = ts.create_mapping(data, early_pats, 2, 4, 32, 'r_random')
test_dict = ts.create_mapping(data, test_pats, 2, np.inf, 32, 'r_random')
train_dict = ts.enrichment_by_ID_list(data, train_pats, 2, random_state=1)

# create data
exlude_features = ['sos', 'aoc', 'aos', 'skewness', 'kurtosis']
print(f'Create Xy train', flush=True)
train_data = ts.temporal_to_static(data, train_dict, temporal_features, constant_features, exlude_features=exlude_features)
print(f'Create Xy val', flush=True)
early_data = ts.temporal_to_static(data, early_stopping_dict, temporal_features, constant_features, exlude_features=exlude_features)
print(f'Create Xy test', flush=True)
test_data = ts.temporal_to_static(data, test_dict, temporal_features, constant_features, exlude_features=exlude_features)

# -------------------------------------------------------------------------- #
# ------------------------------ Fit final model --------------------------- #
# -------------------------------------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1)
y_train = train_data['y']
X_early = early_data.drop(['ID', 'y'], 1)
y_early = early_data['y']
X_test = test_data.drop(['ID', 'y'], 1)
y_test = test_data['y']

# Create model
model = XGBRegressor(max_depth=int(params['max_depth']),
                     learning_rate=params['learning_rate'],
                     subsample=params['subsample'],
                     colsample_bytree=params['colsample_bytree'],
                     colsample_bylevel=params['colsample_bylevel'],
                     min_child_weight=params['min_child_weight'],
                     gamma=params['gamma'],
                     reg_lambda=params['reg_lambda'],
                     n_estimators=7000,
                     objective='reg:squarederror')

# Fit model using early stopping
early = early_stop(stopping_rounds=30, maximize=False)
model.fit(X_train, y_train,
          eval_set=[(X_early, y_early)],
          callbacks=[early])

# -------------------------------------------------------------------------- #
# -------------------------------- Predicitions ----------------------------- #
# -------------------------------------------------------------------------- #

# test predictions
predictions = model.predict(X_test, ntree_limit=model.get_booster().best_iteration)

# save feature importance
feature_importances = pd.DataFrame(dict(feature=train_data.drop(['ID', 'y'], 1).columns.tolist(),
                                        importance=model.feature_importances_))

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID_i in test_data.ID.unique():
    ID, idx = ID_i.split('_')
    x_v, y_v = test_dict[ID][int(idx)]
    summary = summary.append(dict(ID=ID,
                                  xlast=x_v[-1],
                                  t_pred=ts.time_between_records(data, x_v[-1], y_v[-1], ID),
                                  yvis=y_v[-1],
                                  y_true=data.loc[(data['ID'] == ID) & (data['no'] == y_v[0]), 'y'].item(),
                                  y_last=data.loc[(data['ID'] == ID) & (data['no'] == x_v[-1]), 'y'].item()),
                             ignore_index=True)

summary['pred'] = predictions
from sklearn.metrics import mean_squared_error
if mean_squared_error(summary['pred'], summary['y_true']) ** 0.5 == mean_squared_error(predictions, y_test) ** 0.5:
    print("OK")
else:
    print("Something wrong")

# Save predictions
summary.to_csv(summary_csv, index=False)
feature_importances.to_csv(importance_csv, index=False)
model.save_model(MODEL_PATH)

print("Finished successfully")