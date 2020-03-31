import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
import pickle
from Models import LSTM_pytorch as model_lib
import torch

model_name = 'LSTM'

TUNING_PATH = f'C:/Users/benhada/Desktop/Results/PROACT/{model_name}/tuning_files/tuning_0.pkl'
prev_predictions_folder = f'C:/Users/benhada/Desktop/Results/train_PROACT_test_TAMC/{model_name}/predictions/'
new_predictions_folder = f'C:/Users/benhada/Desktop/Results/task_transfer/{model_name}/predictions/'
INITIAL_MODEL_PATH = f'C:/Users/benhada/Desktop/Results/task_transfer/{model_name}/models/initial_model.model'
models_folder = f'C:/Users/benhada/Desktop/Results/task_transfer/{model_name}/models/'
proact = pd.read_csv('C:/Users/benhada/Desktop/Results/train_PROACT_test_TAMC/Data/train.csv')
tamc = pd.read_csv('C:/Users/benhada/Desktop/Results/train_PROACT_test_TAMC/Data/test.csv')

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features

proact = proact[['ID', 'time'] + features + ['y']]
tamc = tamc[['ID', 'time'] + features + ['y']]

ts.drop_by_number_of_records(proact, 3, inplace=True)
ts.add_ordering(proact, inplace=True)
ts.add_ordering(tamc, inplace=True)

# ----------------------------------------- #
# --------------- train-val --------------- #
# ----------------------------------------- #

print(f'>> Split Train-Validation-Test', flush=True)
n_val_samples = int(0.15*proact['ID'].nunique())
train_pats, val_pats = train_test_split(proact['ID'].unique(), test_size=n_val_samples, random_state=32)

# create test dictionary (unique for each iteration)
print(f'>> Split x,y visits for each validation and test patient', flush=True)
val_dict = ts.create_mapping(proact, val_pats, 2, 4, 32)

# enrichment
print(f'>> Training data enrichment', flush=True)
train_dict = ts.enrichment_by_ID_list(proact,
                                      list_IDs=train_pats,
                                      min_x_visits=2,
                                      progress_bar=True,
                                      random_state=1)

# Scaling
features_to_scale = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food',
                     'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS', 'Age',
                     'TimeSinceOnset']

preprocessed_data = ts.scaling_wDictionary(proact, dict(), 'MinMax', features_to_scale)

# create Train-Validation for early stopping
print(f'Create Xy train', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(preprocessed_data, features, train_dict, True,  1, True, t_pred='time_to_next')
print(f'Create Xy validation', flush=True)
X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, True, 1, True, t_pred='time_to_next')

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ---------------------------- #
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(TUNING_PATH, 'rb') as f:
    a = pickle.load(f)
a = a[a.validation_mse == a.validation_mse.min()]
params = dict(zip(a.columns.tolist(), a.values[0].tolist()))

early_stopping = dict(patience=30, verbose=False, delta=0, path=INITIAL_MODEL_PATH)

model = model_lib.LSTMnet(input_size=len(features) + 1,
                          lstm_hidden_units=int(params['hidden_size']),
                          n_lstm_layers=int(params['num_layers']))

model.compile(optimizer='adam', loss='mse', lr=params['lr'])
model.fit(X_train, y_train, lengths_train,
          validation_data=(X_val, y_val, lengths_val),
          early_stopping=early_stopping,
          batch_size=int(params['batch_size']),
          epochs=3000,
          use_cuda=False)

model.load_state_dict(torch.load(INITIAL_MODEL_PATH))

# ------------------------------------------------------------------- #
# ---------------------------- Fit on tamc -------------------------- #
# ------------------------------------------------------------------- #

for number_iter in range(60):
    print("# ------------------------------------------------------------------- #")
    print(f"# ---------------------- {number_iter} / 60 ----------------------------- #")
    print("# ------------------------------------------------------------------- #")
    n_test_samples = int(0.1 * tamc['ID'].nunique())
    # split
    tamc_train_pats, tamc_test_pats = train_test_split(tamc['ID'].unique(), test_size=n_test_samples, random_state=number_iter)
    tamc_train_pats, tamc_val_pats = train_test_split(tamc_train_pats, test_size=n_test_samples, random_state=number_iter)

    # dictionaries
    val_dict = ts.create_mapping(tamc, tamc_val_pats, 2, 4, 32)
    test_dict = ts.create_mapping(tamc, tamc_test_pats, 2, 4, 32)
    train_dict = ts.enrichment_by_ID_list(tamc,
                                          list_IDs=tamc_train_pats,
                                          min_x_visits=2,
                                          progress_bar=True,
                                          random_state=1)

    # scaling
    tamc_scaled = ts.scaling_wDictionary(tamc, test_dict, 'MinMax', features_to_scale)

    # create Xy train, validation and test
    tamc_X_train, tamc_y_train, tamc_lengths_train = ts.create_temporal_Xy_lists(tamc_scaled, features, train_dict, True, 1, True, t_pred='time_to_next')
    tamc_X_val, tamc_y_val, tamc_lengths_val = ts.create_temporal_Xy_lists(tamc_scaled, features, val_dict, True, 1, True, t_pred='time_to_next')
    tamc_X_test, tamc_y_test, tamc_lengths_test = ts.create_temporal_Xy_lists(tamc_scaled, features, test_dict, True, 1, True, t_pred='time_to_next')

    new_model = model_lib.LSTMnet(input_size=len(features) + 1,
                                  lstm_hidden_units=int(params['hidden_size']),
                                  n_lstm_layers=int(params['num_layers']))

    new_model.load_state_dict(torch.load(INITIAL_MODEL_PATH))

    new_model.compile(optimizer='adam', loss='mse', lr=params['lr'])

    early_stopping = dict(patience=30, verbose=False, delta=0, path=models_folder+f'{number_iter}.model')
    new_model.fit(tamc_X_train, tamc_y_train, tamc_lengths_train,
                  validation_data=(tamc_X_val, tamc_y_val, tamc_lengths_val),
                  early_stopping=early_stopping,
                  batch_size=int(params['batch_size']),
                  epochs=3000,
                  use_cuda=False)

    new_model.load_state_dict(torch.load(models_folder+f'{number_iter}.model'))

    # ------------------------------------------------------------------- #
    # ------------------------- Test prediction ------------------------- #
    # ------------------------------------------------------------------- #

    preds = model.predict(tamc_X_test, tamc_lengths_test)
    old_preds = pd.read_csv(prev_predictions_folder + f'{number_iter}.csv')
    old_preds['tt_pred'] = preds
    old_preds.to_csv(new_predictions_folder+f'{number_iter}.csv', index=False)