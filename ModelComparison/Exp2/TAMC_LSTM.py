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

number_iter = 0
folder = 'C:/Users/Ben/Desktop/Results/'
tuning_path = folder + f'TAMC/{model_name}/tuning_files/tuning_0.pkl'
new_preds_path = folder + f'Exp2/TAMC_{model_name}/predictions/'
MODEL_PATH = folder + f'Exp2/TAMC_{model_name}/models/model_{number_iter}.model'
TAMC = pd.read_csv(folder + 'TAMC2.csv')
train_test_file = folder + 'TAMC_2_train_test_split.pkl'

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features

with open(train_test_file, 'rb') as f:
    split = pickle.load(f)
train_pats = split[number_iter]['train_pats']

# ----------------------------------------- #
# --------------- train-val --------------- #
# ----------------------------------------- #

# Split Train-Validation-Test
n_val_samples = int(0.1*len(train_pats))
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=32)

# create dicitonaries
val_dict = ts.create_mapping(TAMC, val_pats, 2, 10, 32, 'n_samples')
train_dict = ts.enrichment_by_ID_list(TAMC, train_pats, 2, random_state=1)

# ----- Scaling -----
features_to_scale = ['time', 'Speech', 'Salivation', 'Swallowing', 'Writing',
                     'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS',
                     'Age', 'TimeSinceOnset']
patients_to_fit_scaler = train_pats.tolist()
scaler = MinMaxScaler((1, 2))
scaler.fit(TAMC.loc[TAMC['ID'].isin(patients_to_fit_scaler), features_to_scale])
preprocessed_data = TAMC.copy()
preprocessed_data.loc[:, features_to_scale] = scaler.transform(preprocessed_data.loc[:, features_to_scale])

# create Train-Validation for early stopping
print('Create data', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(preprocessed_data, features, train_dict, True, 1, True, t_pred=True, time_to_next=True)
X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, True, 1, True, t_pred=True, time_to_next=True)

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ---------------------------- #
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(tuning_path, 'rb') as f:
    a = pickle.load(f)
a = a[a.validation_mse == a.validation_mse.min()]
params = dict(zip(a.columns.tolist(), a.values[0].tolist()))

early_stopping = dict(patience=30, verbose=False, delta=0, path=MODEL_PATH)

model = model_lib.LSTMnet(input_size=len(features) + 2,
                          lstm_hidden_units=int(params['hidden_size']),
                          n_lstm_layers=int(params['num_layers']))

model.compile(optimizer='adam', loss='mse', lr=params['lr'])


model.fit(X_train, y_train, lengths_train,
          validation_data=(X_val, y_val, lengths_val),
          early_stopping=early_stopping,
          batch_size=int(params['batch_size']),
          epochs=2,
          use_cuda=False)

model.load_state_dict(torch.load(MODEL_PATH))

# ------------------------------------------------------------------- #
# ------------------------- Test prediction ------------------------- #
# ------------------------------------------------------------------- #

print(f"--------- {number_iter} --------- ")

TAMC_test_pats = split[number_iter]['test_pats']

test_dict = ts.create_mapping(preprocessed_data, TAMC_test_pats, 2, np.inf, 32, 'n_samples')

X_test, y_test, lengths_test = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, True, 1, True, t_pred=True, time_to_next=True)

predictions = model.predict(X_test, lengths_test, numpy=True)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID in [*test_dict.keys()]:
    for idx in range(len(test_dict[ID])):
        x_v, y_v = test_dict[ID][int(idx)]
        summary = summary.append(dict(ID=ID,
                                      xlast=x_v[-1],
                                      t_pred=ts.time_between_records(TAMC, x_v[-1], y_v[-1], ID),
                                      yvis=y_v[-1],
                                      y_true=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == y_v[0]), 'y'].item(),
                                      y_last=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == x_v[-1]), 'y'].item()),
                                 ignore_index=True)

summary['pred'] = predictions

# save predictions
summary.to_csv(new_preds_path + f'{number_iter}.csv', index=False)
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

number_iter = 0
folder = 'C:/Users/Ben/Desktop/Results/'
tuning_path = folder + f'TAMC/{model_name}/tuning_files/tuning_0.pkl'
new_preds_path = folder + f'Exp2/TAMC_{model_name}/predictions/'
MODEL_PATH = folder + f'Exp2/TAMC_{model_name}/models/model_{number_iter}.model'
TAMC = pd.read_csv(folder + 'TAMC2.csv')
train_test_file = folder + 'TAMC_2_train_test_split.pkl'

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features

with open(train_test_file, 'rb') as f:
    split = pickle.load(f)
train_pats = split[number_iter]['train_pats']

# ----------------------------------------- #
# --------------- train-val --------------- #
# ----------------------------------------- #

# Split Train-Validation-Test
n_val_samples = int(0.1*len(train_pats))
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=32)

# create dicitonaries
val_dict = ts.create_mapping(TAMC, val_pats, 2, 10, 32, 'n_samples')
train_dict = ts.enrichment_by_ID_list(TAMC, train_pats, 2, random_state=1)

# ----- Scaling -----
features_to_scale = ['time', 'Speech', 'Salivation', 'Swallowing', 'Writing',
                     'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS',
                     'Age', 'TimeSinceOnset']
patients_to_fit_scaler = train_pats.tolist()
scaler = MinMaxScaler((1, 2))
scaler.fit(TAMC.loc[TAMC['ID'].isin(patients_to_fit_scaler), features_to_scale])
preprocessed_data = TAMC.copy()
preprocessed_data.loc[:, features_to_scale] = scaler.transform(preprocessed_data.loc[:, features_to_scale])

# create Train-Validation for early stopping
print('Create data', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(preprocessed_data, features, train_dict, True, 1, True, t_pred=True, time_to_next=True)
X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, True, 1, True, t_pred=True, time_to_next=True)

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ---------------------------- #
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(tuning_path, 'rb') as f:
    a = pickle.load(f)
a = a[a.validation_mse == a.validation_mse.min()]
params = dict(zip(a.columns.tolist(), a.values[0].tolist()))

early_stopping = dict(patience=30, verbose=False, delta=0, path=MODEL_PATH)

model = model_lib.LSTMnet(input_size=len(features) + 2,
                          lstm_hidden_units=int(params['hidden_size']),
                          n_lstm_layers=int(params['num_layers']))

model.compile(optimizer='adam', loss='mse', lr=params['lr'])


model.fit(X_train, y_train, lengths_train,
          validation_data=(X_val, y_val, lengths_val),
          early_stopping=early_stopping,
          batch_size=int(params['batch_size']),
          epochs=2,
          use_cuda=False)

model.load_state_dict(torch.load(MODEL_PATH))

# ------------------------------------------------------------------- #
# ------------------------- Test prediction ------------------------- #
# ------------------------------------------------------------------- #

print(f"--------- {number_iter} --------- ")

TAMC_test_pats = split[number_iter]['test_pats']

test_dict = ts.create_mapping(preprocessed_data, TAMC_test_pats, 2, np.inf, 32, 'n_samples')

X_test, y_test, lengths_test = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, True, 1, True, t_pred=True, time_to_next=True)

predictions = model.predict(X_test, lengths_test, numpy=True)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID in [*test_dict.keys()]:
    for idx in range(len(test_dict[ID])):
        x_v, y_v = test_dict[ID][int(idx)]
        summary = summary.append(dict(ID=ID,
                                      xlast=x_v[-1],
                                      t_pred=ts.time_between_records(TAMC, x_v[-1], y_v[-1], ID),
                                      yvis=y_v[-1],
                                      y_true=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == y_v[0]), 'y'].item(),
                                      y_last=TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == x_v[-1]), 'y'].item()),
                                 ignore_index=True)

summary['pred'] = predictions

# save predictions
summary.to_csv(new_preds_path + f'{number_iter}.csv', index=False)