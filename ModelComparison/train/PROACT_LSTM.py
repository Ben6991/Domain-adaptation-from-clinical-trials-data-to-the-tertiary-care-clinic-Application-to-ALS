import sys
import os

# os.chdir('')
# sys.path.append('')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error
import torch
import pickle
import numpy as np
from sklearn.model_selection import ParameterSampler

# for number_iter in range(64, 100):
number_iter = 0
data_file = f'C:/Users/Ben/Desktop/Data/PROACT/data.csv'
MODEL_PATH = f'C:/Users/Ben/Desktop/model_{number_iter}.model'
TUNING_PATH = f'C:/Users/Ben/master/ModelComparison/tuning_{number_iter}.pkl'
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
features = temporal_numeric_features + constant_features  # define features

# drop patients with less than 3 visits
print(f'>> Drop patients with less than 3 visits', flush=True)
ts.drop_by_number_of_records(data, 3, inplace=True, verbose=2)
ts.add_ordering(data, inplace=True)

# split patients into train-validation-test (unique for each iteration)
print(f'>> Split Train-Validation-Test', flush=True)
n_test_samples = int(0.15*data['ID'].nunique())
n_val_samples = int(0.15*data['ID'].nunique())
n_early_stopping_samples = int(0.1*data['ID'].nunique())
train_pats, test_pats = train_test_split(data['ID'].unique(), test_size=n_test_samples, random_state=number_iter) ## was 32
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=number_iter)
train_pats, early_stop_pats = train_test_split(train_pats, test_size=n_early_stopping_samples, random_state=number_iter)

# create test dictionary (unique for each iteration)
print(f'>> Split x,y visits for each validation and test patient', flush=True)
test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random', target)
val_dict = ts.create_mapping(data, val_pats, 2, 4, 32, 'r_random', target)
early_stopping_dict = ts.create_mapping(data, early_stop_pats, 2, 4, 32, 'r_random', target)


# ------------------------------ Hyper-parameter tuning ------------------------------

# enrichment
print(f'>> Training data enrichment...', flush=True)
train_dict = ts.enrichment_by_ID_list(data,
                                      list_IDs=train_pats,
                                      min_x_visits=2,
                                      target=target,
                                      progress_bar=True)

# Scaling
preprocessed_data = ts.scaling_wDictionary(data[~data['ID'].isin(test_pats)], val_dict, 'MinMax', features_to_scale)


# create Train-Validation for early stopping
print(f'Create Xy train...', flush=True)
X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(preprocessed_data, features, train_dict, torch_tensors, 1, True)
print(f'Create Xy validation...', flush=True)
X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, torch_tensors, 1, True)
print(f'Create Xy for early stopping...', flush=True)
X_early, y_early, lengths_early = ts.create_temporal_Xy_lists(preprocessed_data, features, early_stopping_dict, torch_tensors, 1, True)

# Random search ?? configurations
param_dist = dict(lr=[0.0001, 0.001, 0.01],
                  hidden_size=2**np.arange(3, 11, 1),
                  # dropout=np.arange(0, 0.6, 0.1),
                  batch_size=(2**np.arange(4, 9, 1)).tolist(),
                  num_layers=[1, 1, 2])

import Models.LSTM_pytorch as model_lib
early_stopping = dict(patience=30, verbose=False, delta=0, path=MODEL_PATH)
import importlib
importlib.reload(model_lib)
n_iter = 15
param_sampler = ParameterSampler(param_dist, n_iter, random_state=123)
min_validation_mse = np.inf
tuning_table = pd.DataFrame(columns=[*param_dist.keys()] + ['validation_mse'])
for i, params in enumerate(param_sampler):
    if params['hidden_size'] >= 2**9 and params['num_layers'] == 2:
        params['hidden_size'] = 2**7

    print(f"----------------------------------------------------------- ")
    print(f"------ Random search iteration number {i+1}/{n_iter} ------ ")
    print(f"----------------------------------------------------------- ")
    ts.print_configuration(params)

    model = model_lib.LSTMnet(input_size=len(features) + 1,
                              lstm_hidden_units=int(params['hidden_size']),
                              n_lstm_layers=int(params['num_layers']))

    model.compile(optimizer='adam', loss='mse', lr=params['lr'])
    model.fit(X_train, y_train, lengths_train,
              validation_data=(X_early, y_early, lengths_early),
              early_stopping=early_stopping,
              batch_size=int(params['batch_size']),
              epochs=1000,
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


