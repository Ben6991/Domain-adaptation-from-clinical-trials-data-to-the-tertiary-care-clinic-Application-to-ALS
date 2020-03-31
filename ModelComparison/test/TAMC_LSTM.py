import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
import pickle
import Models.LSTM_pytorch as model_lib
import torch
from sklearn.preprocessing import MinMaxScaler


number_iter = 0
database = 'TAMC'
model_name = 'LSTM'

folder = 'C:/Users/benhada/Desktop/Results/'
data_file = folder + 'TAMC1.csv'
MODEL_PATH = folder + f'{database}/{model_name}/models/{number_iter}.model'
train_test_split_info = folder + 'TAMC_1_train_test_split.pkl'
tuning_file = folder + f'{database}/{model_name}/tuning_files/tuning_{number_iter}.pkl'
summary_csv = folder + f'{database}/{model_name}/predictions/{number_iter}.csv'
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

features_to_scale = temporal_features + ['AgeOnset', 'Age', 'TimeSinceOnset']
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
train_dict = ts.enrichment_by_ID_list(data, train_pats, 2, random_state=1)
test_dict = ts.create_mapping(data, test_pats, 2, np.inf, 32, 'r_random')

# Scaling
scaler = MinMaxScaler((1, 2))
features_to_scale = ['RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct', 'RFingerExt',
                     'LArmAbduct', 'LElbowFlex', 'LElbowExt', 'LFingerAbduct', 'LFingerExt',
                     'RThighFlex', 'RKneeExt', 'RAnkleDorsi', 'RAnklePlantar', 'LThighFlex',
                     'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar', 'Speech', 'Salivation',
                     'Swallowing', 'Writing', 'Food', 'Dressing', 'Turning', 'Walking', 'Stairs',
                     'Dyspnea', 'ALSFRS', 'AgeOnset', 'TimeSinceOnset', 'Age', 'time']

scaled_data = data.copy()
scaler.fit(data[data['ID'].isin(train_pats)][features_to_scale])
scaled_data[features_to_scale] = scaler.transform(scaled_data[features_to_scale])

# create Train-Validation for early stopping
print(f'Create Xy train', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(scaled_data, features, train_dict, True, 1, True, t_pred=True, time_to_next=True)
print(f'Create Xy for early stopping', flush=True)
X_early, y_early, lengths_early = ts.create_temporal_Xy_lists(scaled_data, features, early_stopping_dict, True, 1, True, t_pred=True, time_to_next=True)
print(f'Create Xy validation', flush=True)
X_test, y_test, lengths_test = ts.create_temporal_Xy_lists(scaled_data, features, test_dict, True, 1, True, t_pred=True, time_to_next=True)


# ------------------------------------------------------------------- #
# ---------------------------- Fit model ---------------------------- #
# ------------------------------------------------------------------- #

early_stopping = dict(patience=30, verbose=False, delta=0, path=MODEL_PATH)

model = model_lib.LSTMnet(input_size=len(features),
                          lstm_hidden_units=int(params['hidden_size']),
                          n_lstm_layers=int(params['num_layers']))

model.compile(optimizer='adam', loss='mse', lr=params['lr'])
model.fit(X_train, y_train, lengths_train,
          validation_data=(X_early, y_early, lengths_early),
          early_stopping=early_stopping,
          batch_size=int(params['batch_size']),
          epochs=5000,
          use_cuda=False)
model.load_state_dict(torch.load(MODEL_PATH))


# ------------------------------------------------------------------- #
# --------------------------- Predictions --------------------------- #
# ------------------------------------------------------------------- #

predictions = model.predict(X_test, lengths_test, numpy=True)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID in [*test_dict.keys()]:
    for idx in range(len(test_dict[ID])):
        x_v, y_v = test_dict[ID][int(idx)]
        summary = summary.append(dict(ID=ID,
                                      xlast=x_v[-1],
                                      t_pred=ts.time_between_records(data, x_v[-1], y_v[-1], ID),
                                      yvis=y_v[-1],
                                      y_true=data.loc[(data['ID'] == ID) & (data['no'] == y_v[0]), 'y'].item(),
                                      y_last=data.loc[(data['ID'] == ID) & (data['no'] == x_v[-1]), 'y'].item()),
                                 ignore_index=True)

summary['pred'] = predictions

# save predictions
summary.to_csv(summary_csv, index=False)
