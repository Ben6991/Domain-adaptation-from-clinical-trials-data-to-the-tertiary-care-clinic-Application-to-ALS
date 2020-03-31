import sys
import os

sys.path.append('/gpfs0/boaz/users/benhada/home/Master/')
sys.path.append('/gpfs0/boaz/users/benhada/home/Master/Models')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestRegressor

number_iter = 24
database_name = 'PROACT'
model_name = 'RF'

# ------------------------------------------------------------------ #
# --------------------------Pre-processing --------------------------#
# ------------------------------------------------------------------ #

data_file = f'C:/Users/Ben/Desktop/Data/PROACT/data.csv'
MODEL_PATH = f'C:/Users/Ben/Desktop/Results/{database_name}/{model_name}/models/model_{number_iter}.model'
TUNING_PATH = f'C:/Users/Ben/Desktop/Results/{database_name}/{model_name}/tuning_files/tuning_{number_iter}.pkl'
summary_csv = f'C:/Users/Ben/Desktop/Results/{database_name}/{model_name}/predictions/{number_iter}.csv'
data = pd.read_csv(data_file)
target = 'ALSFRS_Total'
data['y'] = data[target]

data.drop(['mouth', 'hands', 'trunk', 'leg', 'respiratory'], 1, inplace=True)

print(f'>> Drop patients with less than 3 visits', flush=True)
ts.drop_by_number_of_records(data, 3, inplace=True, verbose=2)
ts.add_ordering(data, inplace=True)

# ----------------------------------------- #
# ------------ Define features ------------ #
# ----------------------------------------- #

temporal_features = ['ALSFRS_Total', 'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing',
                     'Q4_Handwriting', 'Q5_Cutting', 'Q6_Dressing_and_Hygiene',
                     'Q7_Turning_in_Bed', 'Q8_Walking', 'Q9_Climbing_Stairs',
                     'Q10_Respiratory', 'FVC', 'CK', 'Creatinine', 'Phosphorus',
                     'Alkaline.Phosphatase', 'Albumin', 'Bilirubin', 'Chloride',
                     'Hematocrit', 'Hemoglobin', 'Potassium', 'Protein', 'Glucose',
                     'Calcium', 'Sodium', 'Platelets']

constant_features = ['Age', 'Onset.time', 'Onset.Bulbar', 'Onset.Limb', 'Sex.Male']

features = temporal_features + constant_features

# Drop all missing values
ts.drop_by_number_of_records(data, 3, inplace=True)
ts.add_ordering(data, inplace=True)

data = data[['ID', 'time', 'no'] + features + ['y']]
data.reset_index(drop=True, inplace=True)

# ------------------------------------------ #
# ------------ Split train-test ------------ #
# ------------------------------------------ #

print(f'>> Split Train-Validation-Test', flush=True)
n_test_samples = int(0.15*data['ID'].nunique())
train_pats, test_pats = train_test_split(data['ID'].unique(), test_size=n_test_samples, random_state=number_iter)

# create test dictionary (unique for each iteration)
print(f'>> Split x,y visits for each validation and test patient', flush=True)
test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random')

# ------------------------------------------------------------------------------------ #
# ------------------------------ Hyper-parameter tuning ------------------------------ #
# ------------------------------------------------------------------------------------ #

# enrichment
print(f'>> Training data enrichment', flush=True)
train_dict = ts.enrichment_by_ID_list(data,
                                      list_IDs=train_pats,
                                      min_x_visits=2,
                                      progress_bar=True)


# create Train-Validation for early stopping
print(f'Create Xy train', flush=True)
train_data = ts.temporal_to_static(data, train_dict, temporal_features, constant_features)

print(f'Create Xy test', flush=True)
test_data = ts.temporal_to_static(data, test_dict, temporal_features, constant_features)


kurtosis_features = [x for x in train_data.columns if len(x) > 4 and x[-4:] == 'kurt']
skewness_features = [x for x in train_data.columns if len(x) > 4 and x[-4:] == 'skew']

# Fill na (RandomForest only)
train_data[kurtosis_features + skewness_features] = train_data[kurtosis_features + skewness_features].fillna(0)
test_data[kurtosis_features + skewness_features] = test_data[kurtosis_features + skewness_features].fillna(0)

# ----------------------------------------------- #
# ---------------- Random search ---------------- #
# ----------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_test = test_data.drop(['ID', 'y'], 1).values
y_test = test_data['y'].values

# ------------------------------------------------------------------- #
# ---------------------------- Fit model ----------------------------#
# ------------------------------------------------------------------- #

# Get best hyper-parameters
with open(TUNING_PATH, 'rb') as f:
    a = pickle.load(f)
a = a[a.validation_mse == a.validation_mse.min()]
params = dict(zip(a.columns.tolist(), a.values[0].tolist()))

# Create model
model = RandomForestRegressor(n_estimators=params['n_estimators'],
                              max_depth=params['max_depth'],
                              max_features=params['max_features'],
                              min_samples_leaf=params['min_samples_leaf'],
                              min_samples_split=params['min_samples_split'],
                              bootstrap=params['bootstrap'])

# Fit model using early stopping
model.fit(X_train, y_train)

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

# Test prediction
predictions = model.predict(X_test)

summary = pd.DataFrame(columns=['ID', 'xlast', 'yvis', 'y_last', 'y_true', 't_pred'])
for ID_i in test_data.ID.unique():
    ID, idx = ID_i.split('_')
    x_v, y_v = test_dict[int(ID)][int(idx)]
    summary = summary.append(dict(ID=ID,
                                  xlast=x_v[-1],
                                  t_pred=ts.time_between_records(data, x_v[-1], y_v[-1], int(ID)),
                                  yvis=y_v[-1],
                                  y_true=data.loc[(data['ID'] == int(ID)) & (data['no'] == y_v[0]), 'y'].item(),
                                  y_last=data.loc[(data['ID'] == int(ID)) & (data['no'] == x_v[-1]), 'y'].item()),
                             ignore_index=True)

summary['pred'] = predictions

summary.to_csv(summary_csv, index=False)


