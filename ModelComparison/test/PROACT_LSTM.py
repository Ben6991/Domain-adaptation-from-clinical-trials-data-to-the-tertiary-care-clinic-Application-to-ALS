# TEST

import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error
import torch
import pickle

number_iter = 0
database_name = 'PROACT'
model_name = 'RF'

# ------------------------------------------------------------------ #
# --------------------------Pre-processing --------------------------#
# ------------------------------------------------------------------ #

data_file = f'C:/Users/benhada/Desktop/Data/PROACT/data.csv'
MODEL_PATH = f'C:/Users/benhada/Desktop/Results/{database_name}/{model_name}/models/model_{number_iter}.model'
TUNING_PATH = f'C:/Users/benhada/Desktop/Results/{database_name}/{model_name}/tuning_files/tuning_{number_iter}.pkl'
summary_csv = f'C:/Users/benhada/Desktop/Results/{database_name}/{model_name}/predictions/{number_iter}.csv'
data = pd.read_csv(data_file)
target = 'ALSFRS_Total'
data['y'] = data[target]
torch_tensors = True

FRS = ['Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing',
       'Q4_Handwriting', 'Q5_Cutting', 'Q6_Dressing_and_Hygiene',
       'Q7_Turning_in_Bed', 'Q8_Walking', 'Q9_Climbing_Stairs',
       'Q10_Respiratory', 'ALSFRS_Total']

lab_test = ['FVC', 'CK', 'Creatinine', 'Phosphorus',
            'Alkaline.Phosphatase', 'Albumin', 'Bilirubin', 'Chloride',
            'Hematocrit', 'Hemoglobin', 'Potassium', 'Protein', 'Glucose',
            'Calcium', 'Sodium', 'Platelets']

features_to_scale = FRS + lab_test + ['time']

temporal_numeric_features = FRS + lab_test

constant_features = ['Onset.Bulbar', 'Onset.Limb', 'Sex.Male', 'Age']
features = temporal_numeric_features + constant_features

# drop patients with less than 3 visits
print(f'>> Drop patients with less than 3 visits', flush=True)
ts.drop_by_number_of_records(data, 3, inplace=True, verbose=2)
ts.add_ordering(data, inplace=True)

# split patients into train-validation-test (unique for each iteration)
print(f'>> Split Train-Validation-Test', flush=True)
n_test_samples = int(0.15*data['ID'].nunique())
n_val_samples = int(0.15*data['ID'].nunique())
train_pats, test_pats = train_test_split(data['ID'].unique(), test_size=n_test_samples, random_state=number_iter) ## was 32
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=number_iter)

# create test dictionary (unique for each iteration)
print(f'>> Split x,y visits for each validation and test patient', flush=True)
test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random', target)
val_dict = ts.create_mapping(data, val_pats, 2, 4, 32, 'r_random', target)
print(f'>> Training data enrichment', flush=True)
train_dict = ts.enrichment_by_ID_list(data,
                                      list_IDs=train_pats,
                                      min_x_visits=2,
                                      target=target,
                                      progress_bar=True)

# Scaling
preprocessed_data = ts.scaling_wDictionary(data, test_dict, 'MinMax', features_to_scale)

# create Train-Validation for early stopping
print(f'Create Xy train', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(preprocessed_data, features, train_dict, torch_tensors, 1, True)
print(f'Create Xy validation', flush=True)
X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, torch_tensors, 1, True)
print(f'Create Xy  test', flush=True)
X_test, y_test, lengths_test = ts.create_temporal_Xy_lists(preprocessed_data, features, test_dict, torch_tensors, 1, True)

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ----------------------------#
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(TUNING_PATH, 'rb') as f:
    a = pickle.load(f)
a = a[a.validation_mse == a.validation_mse.min()]
params = dict(zip(a.columns.tolist(), a.values[0].tolist()))

# Create and fit model
import Models.LSTM_pytorch as model_lib
early_stopping = dict(patience=30, verbose=False, delta=0, path=MODEL_PATH)

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

predictions = model.predict(X_test, lengths_test, numpy=True)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID in [*test_dict.keys()]:
    for idx in range(len(test_dict[ID])):
        x_v, y_v = test_dict[ID][int(idx)]
        summary = summary.append(dict(ID=ID,
                                      xlast=x_v[-1],
                                      t_pred = ts.time_between_records(data, x_v[-1], y_v[-1], ID),
                                      yvis=y_v[-1],
                                      y_true=data.loc[(data['ID'] == ID) & (data['no'] == y_v[0]), 'y'].item(),
                                      y_last=data.loc[(data['ID'] == ID) & (data['no'] == x_v[-1]), 'y'].item()),
                                 ignore_index=True)


summary['pred'] = predictions

summary.to_csv(summary_csv, index=False)
