import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
import pickle
from sklearn.ensemble import RandomForestRegressor

number_iter = 0
database = 'TAMC'
model_name = 'RF'

folder = 'C:/Users/benhada/Desktop/Results/'
data_file = folder + f'TAMC1.csv'
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

# ----------------------------------------- #
# ---------- Get Best parameters ---------- #
# ----------------------------------------- #

with open(tuning_file, 'rb') as f:
    tuning_table = pickle.load(f)
values = tuning_table.loc[np.argmin(tuning_table['validation_mse'])].values.tolist()
parameters = tuning_table.columns.tolist()
params = dict(zip(parameters, values))

# ------------------------------------------------ #
# ------------ Split train-validation ------------ #
# ------------------------------------------------ #

with open(train_test_split_info, 'rb') as f:
    split = pickle.load(f)
train_pats = split[number_iter]['train_pats']
test_pats = split[number_iter]['test_pats']

# create test dictionary (unique for each iteration)
print(f'Create dictionaries', flush=True)
train_dict = ts.enrichment_by_ID_list(data, train_pats, 2, random_state=1)
test_dict = ts.create_mapping(data, test_pats, 2, np.inf, 32)

# --------------------------------------------------------- #
# ---------------- Create data for training --------------- #
# --------------------------------------------------------- #

# create Train-Validation for early stopping
exlude_features = ['sos', 'aoc', 'aos', 'skewness', 'kurtosis']
print(f'Create Xy train', flush=True)
train_data = ts.temporal_to_static(data, train_dict, temporal_features, constant_features, exlude_features=exlude_features)
print(f'Create Xy test', flush=True)
test_data = ts.temporal_to_static(data, test_dict, temporal_features, constant_features, exlude_features=exlude_features)

# -------------------------------------------------------------------------- #
# ------------------------------ Fit final model --------------------------- #
# -------------------------------------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_test = test_data.drop(['ID', 'y'], 1).values
y_test = test_data['y'].values

model = RandomForestRegressor(n_estimators=params['n_estimators'],
                              max_depth=params['max_depth'],
                              max_features=params['max_features'],
                              min_samples_leaf=params['min_samples_leaf'],
                              min_samples_split=params['min_samples_split'],
                              bootstrap=params['bootstrap'])

# Fit model using early stopping
model.fit(X_train, y_train)

# -------------------------------------------------------------------------- #
# -------------------------------- Predicitons ----------------------------- #
# -------------------------------------------------------------------------- #

# Test prediction
predictions = model.predict(X_test)

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

# Save predictions
summary.to_csv(summary_csv, index=False)

# Save feature importance
feature_importances.to_csv(importance_csv, index=False)

# Save model
with open(MODEL_PATH, 'wb') as f:
     pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

print("Finished successfully")