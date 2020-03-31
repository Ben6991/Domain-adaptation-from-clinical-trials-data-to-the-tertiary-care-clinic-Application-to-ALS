import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
import pickle

data_file = f'C:/Users/benhada/Desktop/Data/Ichilov/072019_.csv'
data = pd.read_csv(data_file)
target = 'ALSFRS'
data['y'] = data[target]

# create 'Age' in TAMC database
for i in range(data.shape[0]):
    if pd.notnull(data.loc[i, 'BirthYear']):
        data.loc[i, 'Age'] = int(data.loc[i, 'VisitDate'].split('-')[0]) - int(data.loc[i, 'BirthYear'])

MMT = ['RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
       'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt',
       'LFingerAbduct', 'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi',
       'RAnklePlantar', 'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar']

FRS = ['Speech', 'Salivation', 'Swallowing',
       'Writing', 'Food', 'Dressing', 'Hands', 'Turning', 'Walking', 'Stairs',
       'Dyspnea']

FRS_TOTAL = ['ALSFRS']

temporal_numeric_features = MMT + FRS + FRS_TOTAL

constant_features = ['Age', 'Gender', 'Smoking', 'Education', 'Sport', 'OnsetSite',
                     'DiagnosticDelay', 'Familial', 'Dementia', 'TimeSinceOnset']

features = temporal_numeric_features + constant_features

data = data[['ID', 'time'] + features + ['y']]

print(f'>> Drop rows with more than 50% missing', flush=True)
data = data[data.isna().mean(1) < 0.50]

print(f'>> Drop patients with less than 3 visits', flush=True)
ts.drop_by_number_of_records(data, 3, inplace=True, verbose=2)
ts.add_ordering(data, inplace=True)

# Apply fill in rules
data = ts.backward_imputation(data, MMT, only_if_max=5)
data = ts.backward_imputation(data, FRS, only_if_max=4)
data = ts.backward_imputation(data, FRS_TOTAL, only_if_max=40)
data = ts.forward_imputation(data, MMT + FRS + FRS_TOTAL, only_if_min=0)

# -------------------------------------------------------------------------- #
# ------------------ Handle with the categorical features ------------------ #
# -------------------------------------------------------------------------- #
# isSmoking instead of Smoking
data.loc[:, 'isSmoking'] = np.nan
data.loc[data['Smoking'] > 0, 'isSmoking'] = 1
data.loc[data['Smoking'] == 0, 'isSmoking'] = -1
data.drop('Smoking', 1, inplace=True)
# Familial, Dementia, Sport, OnsetSite, Education, Gender
data.loc[data['Familial'] == 0, 'Familial'] = -1
data.loc[data['Dementia'] == 0, 'Dementia'] = -1
data.loc[data['Sport'] == 0, 'Sport'] = -1
data.loc[data['OnsetSite'] == 0, 'OnsetSite'] = -1
data.loc[data['Education'] == 1, 'Eduction'] = -1
data.loc[data['Education'] == 2, 'Eduction'] = 1
data.loc[data['Gender'] == 0, 'Gender'] = -1
# Drop DiagnosticDelay
data.drop('DiagnosticDelay', 1, inplace=True)

ts.add_missing_indicator(data, 'isSmoking', inplace=True)
ts.add_missing_indicator(data, 'Familial', inplace=True)
ts.add_missing_indicator(data, 'Dementia', inplace=True)
ts.add_missing_indicator(data, 'Sport', inplace=True)
ts.add_missing_indicator(data, 'OnsetSite', inplace=True)
ts.add_missing_indicator(data, 'Education', inplace=True)
ts.add_missing_indicator(data, 'Gender', inplace=True)

data['ALSFRS'] = data[FRS].sum(1, skipna=False)

data.dropna(inplace=True)

ts.drop_by_number_of_records(data, 3, inplace=True)
data.reset_index(drop=True, inplace=True)
ts.add_ordering(data, inplace=True)

# ------------------------------ Hyper-parameter tuning ------------------------------


for number_iter in range(99):
    print(f"-----------------------------------------")
    print(f"----------------- {number_iter}/100 -----------------")
    print(f"-----------------------------------------")

    # split patients into train-validation-test (unique for each iteration)
    # print(f'>> Split Train-Validation-Test', flush=True)
    n_test_samples = int(0.1*data['ID'].nunique())
    n_val_samples = int(0.1*data['ID'].nunique())
    n_early_stopping_samples = int(0.1*data['ID'].nunique())
    train_pats, test_pats = train_test_split(data['ID'].unique(), test_size=n_test_samples, random_state=number_iter) ## was 32
    train_pats, val_pats = train_test_split(train_pats, test_size=n_val_samples, random_state=number_iter)
    train_pats, early_stop_pats = train_test_split(train_pats, test_size=n_early_stopping_samples, random_state=number_iter)

    # create test dictionary (unique for each iteration)
    # print(f'>> Split x,y visits for each validation and test patient', flush=True)
    test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random', target)
    val_dict = ts.create_mapping(data, val_pats, 2, 4, 32, 'r_random', target)
    early_stopping_dict = ts.create_mapping(data, early_stop_pats, 2, 4, 32, 'r_random', target)

    # enrichment
    # print(f'>> Training data enrichment...', flush=True)
    train_dict = ts.enrichment_by_ID_list(data,
                                          list_IDs=train_pats,
                                          min_x_visits=2,
                                          target=target,
                                          progress_bar=True)

    # Scaling
    features_to_scale = ['time',
                         'RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
                         'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt', 'LFingerAbduct',
                         'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi', 'RAnklePlantar',
                         'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar',
                         'Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing',
                         'Turning', 'Walking', 'Stairs', 'Dyspnea',
                         'ALSFRS', 'Age', 'TimeSinceOnset']

    preprocessed_data = ts.scaling_wDictionary(data[~data['ID'].isin(test_pats)], val_dict, 'MinMax', features_to_scale)


    features = ['RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
                'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt', 'LFingerAbduct',
                'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi', 'RAnklePlantar',
                'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar', 'Speech',
                'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing',
                'Turning', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS', 'Age', 'Gender',
                'Education', 'Sport', 'OnsetSite', 'Familial', 'Dementia',
                'TimeSinceOnset', 'isSmoking', 'Eduction', 'isSmokingNA',
                'FamilialNA', 'DementiaNA', 'SportNA', 'OnsetSiteNA', 'EducationNA',
                'GenderNA']

    # 46 features + time prediction

    # create Train-Validation for early stopping
    # print(f'Create Xy train...', flush=True)
    X_train, y_train, lengths_train = ts.create_temporal_Xy_lists(preprocessed_data, features, train_dict, torch_tensors, 1, True)
    # print(f'Create Xy validation...', flush=True)
    X_val, y_val, lengths_val = ts.create_temporal_Xy_lists(preprocessed_data, features, val_dict, torch_tensors, 1, True)
    # print(f'Create Xy for early stopping...', flush=True)
    X_early, y_early, lengths_early = ts.create_temporal_Xy_lists(preprocessed_data, features, early_stopping_dict, torch_tensors, 1, True)
    # print(f'Create Xy for early stopping...', flush=True)
    X_test, y_test, length_test = ts.create_temporal_Xy_lists(preprocessed_data, features, early_stopping_dict, torch_tensors, 1, True)

    to_save = dict(train=[X_train, y_train, lengths_train],
                   validation =[X_val, y_val, lengths_val],
                   early=[X_early, y_early, lengths_train],
                   test=[X_test, y_test, lengths_train])

    with open(f'C:/Users/Ben/Desktop/TAMC_data/{number_iter}.pkl', 'wb') as f:
        pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)



