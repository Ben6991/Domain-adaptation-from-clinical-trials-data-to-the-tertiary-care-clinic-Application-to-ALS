import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import TSFunctions as ts
import pickle


data_file = f'C:/Users/Ben/Desktop/072019.csv'
save_to = 'C:/Users/Ben/Desktop/Results/train_PROACT_test_TAMC/Data/test.csv'

# ----------------------------------------- #
# --------------- Load data --------------- #
# ----------------------------------------- #

data = pd.read_csv(data_file)
target = 'ALSFRS'
data['y'] = data[target]

# make TimeSinceInset constant and Age
for ID in data.ID.unique():
    time_since_onset = data.loc[data['ID'] == ID, 'TimeSinceOnset'].min()
    data.loc[data['ID'] == ID, 'TimeSinceOnset'] = time_since_onset
    birth_year = data.loc[data['ID'] == ID, 'BirthYear'].values[0]
    year = int(data.loc[data['ID'] == ID, 'VisitDate'].values[0].split('-')[0])
    age_ID = year - birth_year
    data.loc[data['ID'] == ID, 'Age'] = age_ID

data.drop(['DiagnosticDelay', 'Education', 'Height', 'Weight', 'FVC',
           'ALSFRSR', 'Legs', 'Hands', 'Respiratory', 'Status',
           'VisitDate', 'OnsetDate',
           'BirthYear'], 1, inplace=True)


print(f'>> Drop patients with less than 3 visits', flush=True)
ts.drop_by_number_of_records(data, 3, inplace=True, verbose=2)
ts.add_ordering(data, inplace=True)

MMT = ['RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
       'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt',
       'LFingerAbduct', 'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi',
       'RAnklePlantar', 'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar']

FRS = ['Speech', 'Salivation', 'Swallowing',
       'Writing', 'Food', 'Dressing', 'Turning',
       'Walking', 'Stairs', 'Dyspnea']

FRS_TOTAL = ['ALSFRS']

# --------------------------------------------------------- #
# ---------------- Missing data imputation ---------------- #
# --------------------------------------------------------- #

data = ts.backward_imputation(data, MMT, only_if_max=5)
data = ts.backward_imputation(data, FRS, only_if_max=4)
data = ts.backward_imputation(data, FRS_TOTAL, only_if_max=40)
data = ts.forward_imputation(data, MMT + FRS + FRS_TOTAL, only_if_min=0)
data = ts.simple_average_interpolation_imputation(data, MMT+FRS+FRS_TOTAL, only_if_equals=True)

data.loc[:, 'isSmoking'] = np.nan
data.loc[data['Smoking'] > 0, 'isSmoking'] = 1
data.loc[data['Smoking'] == 0, 'isSmoking'] = -1
data.drop('Smoking', 1, inplace=True)
data.loc[data['Familial'] == 0, 'Familial'] = -1
data.loc[data['Dementia'] == 0, 'Dementia'] = -1
data.loc[data['Sport'] == 0, 'Sport'] = -1
data.loc[data['OnsetSite'] == 0, 'OnsetSite'] = -1
data.loc[data['Gender'] == 0, 'Gender'] = -1

ts.add_missing_indicator(data, 'isSmoking', inplace=True)
ts.add_missing_indicator(data, 'Familial', inplace=True)
ts.add_missing_indicator(data, 'Dementia', inplace=True)
ts.add_missing_indicator(data, 'Sport', inplace=True)
ts.add_missing_indicator(data, 'OnsetSite', inplace=True)
ts.add_missing_indicator(data, 'Gender', inplace=True)

temporal_features = MMT + FRS + FRS_TOTAL

constant_features = ['AgeOnset', 'Gender', 'GenderNA', 'Sport', 'SportNA',
                     'OnsetSite', 'OnsetSiteNA', 'Familial', 'FamilialNA',
                     'Dementia', 'DementiaNA', 'TimeSinceOnset', 'Age',
                     'isSmoking', 'isSmokingNA']

features = temporal_features + constant_features

# Drop all missing values
data.dropna(subset=features, inplace=True)
data = data[data['time'] <= 24*365/12]
ts.drop_by_number_of_records(data, 3, inplace=True)
ts.add_ordering(data, inplace=True)

data = data[['ID', 'time', 'no'] + features + ['y']]
data.reset_index(drop=True, inplace=True)

# # split patients into train-validation-test (unique for each iteration)
# print(f'>> Split Train-Validation-Test', flush=True)
# n_test_samples = int(0.1 * data['ID'].nunique())
# train_pats, test_pats = train_test_split(data['ID'].unique(), test_size=n_test_samples, random_state=number_iter)
#
# # create test dictionary (unique for each iteration)
# print(f'>> Split x,y visits for each validation and test patient', flush=True)
# test_dict = ts.create_mapping(data, test_pats, 2, 4, 32, 'r_random')

# Convert Gender in TAMC to [-1, 1]
data['Gender'] = data['Gender'].apply(lambda x: -1 if x == 0 else 1)

# Convert OnsetSite to Onset.Limb and Onset.Bulbar
data['Onset.Limb'] = data['OnsetSite'].apply(lambda x: 1 if x == 0 else -1)
data['Onset.Bulbar'] = data['OnsetSite'].apply(lambda x: 1 if x == 1 else -1)

data.to_csv(save_to, index=False)