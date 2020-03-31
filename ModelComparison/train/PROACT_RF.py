import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
from sklearn.metrics import mean_squared_error
import torch
import pickle
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestRegressor

number_iter = 0
database_name = 'PROACT'
model_name = 'RF'

data_file = f'C:/Users/Ben/Desktop/Data/PROACT/data.csv'
# TUNING_PATH = f'/gpfs0/boaz/users/benhada/home/Master/ModelComparison/Results/{database_name}/{model_name}/tuning_files/tuning_{number_iter}.pkl'

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
n_val_samples = int(0.15*data['ID'].nunique())
train_pats, test_pats = train_test_split(data['ID'].unique(), test_size=n_test_samples, random_state=number_iter)
train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=number_iter)

# create test dictionary (unique for each iteration)
print(f'>> Split x,y visits for each validation and test patient', flush=True)
test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random')
val_dict = ts.create_mapping(data, val_pats, 2, 4, 32, 'r_random')

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

train_data.isna().sum().sort_values()


print(f'Create Xy validation', flush=True)
validation_data = ts.temporal_to_static(data, val_dict, temporal_features, constant_features)
print(f'Create Xy test', flush=True)
test_data = ts.temporal_to_static(data, test_dict, temporal_features, constant_features)

kurtosis_features = [x for x in train_data.columns if len(x) > 4 and x[-4:] == 'kurt']
skewness_features = [x for x in train_data.columns if len(x) > 4 and x[-4:] == 'skew']

# Fill na (RandomForest only)
train_data[kurtosis_features + skewness_features] = train_data[kurtosis_features + skewness_features].fillna(0)
validation_data[kurtosis_features + skewness_features] = validation_data[kurtosis_features + skewness_features].fillna(0)
test_data[kurtosis_features + skewness_features] = test_data[kurtosis_features + skewness_features].fillna(0)

# ----------------------------------------------- #
# ---------------- Random search ---------------- #
# ----------------------------------------------- #

X_train = train_data.drop(['ID', 'y'], 1).values
y_train = train_data['y'].values
X_val = validation_data.drop(['ID', 'y'], 1).values
y_val = validation_data['y'].values
X_test = test_data.drop(['ID', 'y'], 1).values
y_test = test_data['y'].values

param_dist = {'bootstrap': [True, False],
              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, None],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10],
              'n_estimators': [200, 300, 400, 500, 600, 700, 800, 1000]}

n_iter = 60
param_sampler = ParameterSampler(param_dist, n_iter, random_state=123)
min_validation_mse = np.inf
tuning_table = pd.DataFrame(columns=[*param_dist.keys()] + ['validation_mse'])

for i, params in enumerate(param_sampler):
    print(f"----------------------------------------------------------- ")
    print(f"------ Random search iteration number {i+1}/{n_iter} ------ ")
    print(f"----------------------------------------------------------- ")
    print(params)

    # Create model
    model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                  max_depth=params['max_depth'],
                                  max_features=params['max_features'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  min_samples_split=params['min_samples_split'],
                                  bootstrap=params['bootstrap'])

    # Fit model using early stopping
    model.fit(X_train, y_train)

    # Validation evaluation
    y_val_preds = model.predict(X_val)
    cur_mse = mean_squared_error(y_val_preds, y_val)

    # save best score
    if cur_mse < min_validation_mse:
        min_validation_mse = cur_mse
        print(f"Validation decreased to {min_validation_mse}", flush=True)

    # save configuration
    params['validation_mse'] = cur_mse
    tuning_table = tuning_table.append(params, ignore_index=True)

# Save tuning table
with open(TUNING_PATH, 'wb') as f:
    pickle.dump(tuning_table, f, pickle.HIGHEST_PROTOCOL)


