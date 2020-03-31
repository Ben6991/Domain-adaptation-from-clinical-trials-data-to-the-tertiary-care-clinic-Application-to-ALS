import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
import pickle
from Models import LSTM_pytorch as model_lib
import torch
from sklearn.preprocessing import MinMaxScaler

model_name = 'LSTM'
folder = f'C:/Users/Ben/Desktop/Results/'
tuning_path = folder + f'PROACT/{model_name}/tuning_files/tuning_0.pkl'
initial_model_path = folder + f'Exp3/{model_name}/initial_model.model'
PROACT = pd.read_csv(folder + 'PROACT2.csv')

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features

PROACT = PROACT[['ID', 'time', 'no'] + features + ['y']]

# ----------------------------------------- #
# --------------- train-val --------------- #
# ----------------------------------------- #

# Split Train-Validation-Test
n_val_samples = int(0.15*PROACT['ID'].nunique())
train_pats, val_pats = train_test_split(PROACT.ID.unique(), test_size=n_val_samples, random_state=32)

# create dicitonaries
val_dict = ts.create_mapping(PROACT, val_pats, 2, 10, 32, 'n_samples')
train_dict = ts.enrichment_by_ID_list(PROACT, train_pats, 2, random_state=1)

PROACT.columns
# Scaling
scaler = MinMaxScaler((1, 2))

features_to_scale = ['time', 'Speech', 'Salivation', 'Swallowing', 'Writing',
                     'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS',
                     'Age', 'TimeSinceOnset']

scaled_data = PROACT.copy()
scaler.fit(PROACT[PROACT['ID'].isin(train_pats)][features_to_scale])
scaled_data[features_to_scale] = scaler.transform(scaled_data[features_to_scale])

# create Train-Validation for early stopping
print(f'Create Xy', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(scaled_data, features, train_dict, True, 1, True, t_pred=True, time_to_next=True)
X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(scaled_data, features, val_dict, True, 1, True, t_pred=True, time_to_next=True)

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ---------------------------- #
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(tuning_path, 'rb') as f:
    a = pickle.load(f)
a = a[a.validation_mse == a.validation_mse.min()]
params = dict(zip(a.columns.tolist(), a.values[0].tolist()))

early_stopping = dict(patience=30, verbose=False, delta=0, path=initial_model_path)

model = model_lib.LSTMnet(input_size=len(features) + 2,
                          lstm_hidden_units=int(params['hidden_size']),
                          n_lstm_layers=int(params['num_layers']))

model.compile(optimizer='adam', loss='mse', lr=params['lr'])

model.fit(X_train, y_train, lengths_train,
          validation_data=(X_val, y_val, lengths_val),
          early_stopping=early_stopping,
          batch_size=int(params['batch_size']),
          epochs=7000,
          use_cuda=False)

print("Finished successfully")