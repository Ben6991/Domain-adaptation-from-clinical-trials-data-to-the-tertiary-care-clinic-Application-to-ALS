import sys
import os

sys.path.append('C:/Users/benhada/master/Models/')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error
import torch
import pickle
from sklearn.model_selection import ParameterSampler
import LSTM_pytorch as model_lib

number_iter = 0
database = 'TAMC'
model_name = 'LSTM'

folder = 'C:/Users/benhada/Desktop/Results/'
data_file = folder + 'TAMC1.csv'
MODEL_PATH = folder + f'{database}/{model_name}/models/{number_iter}.model'
train_test_split_info = folder + 'TAMC_1_train_test_split.pkl'
TUNING_PATH = folder + f'{database}/{model_name}/tuning_files/tuning_{number_iter}.pkl'
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
val_dict = ts.create_mapping(data, val_pats, 2, 5000, 32, 'n_samples')
early_stopping_dict = ts.create_mapping(data, early_pats, 2, 4, 32, 'n_samples')
train_dict = ts.enrichment_by_ID_list(data, train_pats, 2, random_state=1)

# ------------------------------------------------------------------------------------ #
# ------------------------------ Hyper-parameter tuning ------------------------------ #
# ------------------------------------------------------------------------------------ #

# ----- Scaling -----
from sklearn.preprocessing import MinMaxScaler
patients_to_fit_scaler = train_pats.tolist() + early_pats.tolist()
scaler = MinMaxScaler()
scaler.fit(data.loc[data['ID'].isin(patients_to_fit_scaler), features_to_scale])
preprocessed_data = data.copy()
preprocessed_data.loc[:, features_to_scale] = scaler.transform(preprocessed_data.loc[:, features_to_scale])

# create Train-Validation for early stopping
print('Create data', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(preprocessed_data, features, train_dict, True, 1, True, t_pred=True, time_to_next=True)
X_early, y_early, lengths_early = ts.create_temporal_Xy_lists(preprocessed_data, features, early_stopping_dict, True, 1, True, t_pred=True, time_to_next=True)
X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, True, 1, True, t_pred=True, time_to_next=True)

# Random search ?? configurations
param_dist = dict(lr=[0.0001, 0.001, 0.01],
                  hidden_size=2**np.arange(3, 10, 1),
                  # dropout=np.arange(0, 0.6, 0.1),
                  batch_size=(2**np.arange(4, 10, 1)).tolist(),
                  num_layers=[1, 1, 1, 1, 2])

early_stopping = dict(patience=30, verbose=False, delta=0, path=MODEL_PATH)
n_iter = 60
param_sampler = ParameterSampler(param_dist, n_iter, random_state=123)
min_validation_mse = np.inf
tuning_table = pd.DataFrame(columns=[*param_dist.keys()] + ['validation_mse'])

for i, params in enumerate(param_sampler):
    if params['hidden_size'] >= 2**8 and params['num_layers'] == 2:
        params['hidden_size'] = 2**7

    print(f"----------------------------------------------------------- ", flush=True)
    print(f"------ Random search iteration number {i+1}/{n_iter} ------ ", flush=True)
    print(f"----------------------------------------------------------- ", flush=True)
    ts.print_configuration(params)

    model = model_lib.LSTMnet(input_size=len(features) + 2,
                              lstm_hidden_units=int(params['hidden_size']),
                              n_lstm_layers=int(params['num_layers']))


    model.compile(optimizer='adam', loss='mse', lr=params['lr'])
    model.fit(X_train, y_train, lengths_train,
              validation_data=(X_early, y_early, lengths_early),
              early_stopping=early_stopping,
              batch_size=int(params['batch_size']),
              epochs=5000,
              use_cuda=False)

    y_val_preds = model.predict(X_val, lengths_val, numpy=True)

    cur_mse = mean_squared_error(y_val_preds, y_val if not torch.is_tensor(y_val[0]) else [x.detach().numpy() for x in y_val])

    if cur_mse < min_validation_mse:
        min_validation_mse = cur_mse
        print(f"Validation decreased to {min_validation_mse}", flush=True)

    params['validation_mse'] = cur_mse
    tuning_table = tuning_table.append(params, ignore_index=True)

# Save tuning table
with open(TUNING_PATH, 'wb') as f:
    pickle.dump(tuning_table, f, pickle.HIGHEST_PROTOCOL)

print("Finished successfully")
