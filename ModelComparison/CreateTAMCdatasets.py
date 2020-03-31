import numpy as np
import pandas as pd
import time
import TSFunctions as ts
from sklearn.model_selection import train_test_split
import random
import pickle

folder = 'C:/Users/Ben/Desktop/Results/'
data_path = f'C:/Users/Ben/Desktop/072019.csv'
exp1_data_path = 'C:/Users/Ben/Desktop/Results/'
exp2_path = 'C:/Users/Ben/Desktop/Results/'


MMT = ['RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
       'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt',
       'LFingerAbduct', 'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi',
       'RAnklePlantar', 'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar']

FRS = ['Speech', 'Salivation', 'Swallowing',
       'Writing', 'Food', 'Dressing', 'Turning',
       'Walking', 'Stairs', 'Dyspnea']

FRS_TOTAL = ['ALSFRS']


# ------------------------ #
# ----- Load data -------- #
# ------------------------ #

TAMC = pd.read_csv(data_path)
TAMC['y'] = TAMC['ALSFRS']

# ------------------------ #
# ---- Pre-processing ---- #
# ------------------------ #

# Drop un-relevant features
TAMC.drop(['OriginCode', 'DiagnosticDelay', 'Status', 'NeckFlex',
           'NeckExt', 'Bulbar', 'Hands', 'Legs', 'Orthopnea', 'Resp',
           'Respiratory', 'FVC', 'Height', 'Weight', 'ALSFRSR'], 1, inplace=True)

'''
            -- Filter patients --
    1. Choose window size (time range)
    2.  2.1. Drop patients with less than 3 visits
        2.2. Drop patients without any possible y visit 
'''

# 1. Choose window size
window_size = 24 * (365/12)
info = dict()
cause_missing_values = 0
cause_number_of_visits = 0

all_pats = TAMC.ID.unique()
for ID in all_pats:
    ID_data = TAMC[TAMC.ID == ID].sort_values('no').reset_index(drop=True)
    max_candidates = ID_data.shape[0]-2
    cur_visit = 1
    not_finished = True
    while cur_visit <= max_candidates and not_finished:
        temp = ID_data[ID_data['no'] >= cur_visit].reset_index(drop=True)
        temp['range'] = temp['time'].diff().fillna(0).cumsum()
        temp = temp[temp['range'] <= window_size]
        if (temp.shape[0] > 2) and (not all(temp.loc[2:, 'y'].isna())):
            info[ID] = temp['no'].values.tolist()
            not_finished = False
        else:
            cur_visit += 1
    if ID not in [*info.keys()]:
        TAMC = TAMC[TAMC['ID'] != ID].reset_index(drop=True)
        if max_candidates < 1:
            cause_number_of_visits += 1
        else:
            cause_missing_values += 1


import matplotlib.pyplot as plt


t = [3, 7, 9, 10]
x1 = [4, 4, 3, 1]
x2 = [5, 3, 1, 0]

(1-4) / (10-3)
(0-5) / (10-3)
plt.figure(figsize=(3.5, 2.5))
plt.plot(t, x1, marker='o', markersize=8, linestyle='--')
plt.plot(t, x2, marker='^', markersize=8, linestyle=':')
plt.legend(['Speech', 'Salivation'])
plt.xlabel("Time")
plt.savefig('ModelComparison/plots/example.png', bbox_inches='tight')
plt.show()

print(f'{cause_missing_values + cause_number_of_visits} patients dropped:', flush=True)
print(f'\t{cause_number_of_visits} because number of visits, flush=True')
print(f'\t{cause_missing_values} because missing values', flush=True)

# shift visits
for ID in [*info.keys()]:
    visits = info[ID]
    if visits[0] > 1:
        time_shifted = TAMC.loc[(TAMC['ID'] == ID) & (TAMC['no'] == visits[0]), 'time'].item()
        TAMC = TAMC[~((TAMC['ID'] == ID) & (~TAMC['no'].isin(visits)))]
        TAMC.loc[TAMC['ID'] == ID, 'time'] = TAMC[(TAMC['ID'] == ID) & (TAMC['no'].isin(visits))].time.diff().cumsum().fillna(0)
        if pd.notnull(TAMC.loc[TAMC['ID'] == ID, 'TimeSinceOnset'].unique()[0]):
            TAMC.loc[TAMC['ID'] == ID, 'TimeSinceOnset'] += time_shifted
        TAMC.loc[TAMC['ID'] == ID, 'no'] = np.arange(1, len(visits)+1)
    else:
        TAMC = TAMC[~((TAMC['ID'] == ID) & (~TAMC['no'].isin(visits)))]

# max_samples_for_patients = 50
# upsampling = 20
#
# all_pats = TAMC.ID.unique()
# train_enriched = ts.enrichment_by_ID_list(TAMC, all_pats, 2,
#                                           progress_bar=True,
#                                           random_sample=max_samples_for_patients,
#                                           random_state=1)

# for ID in all_pats:
#     cur_samples = len(train_enriched[ID])
#     samples = train_enriched[ID].copy()
#     while cur_samples < upsampling:
#         to_add = upsampling - cur_samples
#         if cur_samples <= 0.5*upsampling:
#             samples *= int(np.floor(upsampling/cur_samples))
#             cur_samples = len(samples)
#         else:
#             samples += random.sample(samples, k=to_add)
#             cur_samples = len(samples)
#     train_enriched[ID] = samples
#

# import pickle
# with open(output_path + f'enrichment.pkl', 'wb') as f:
#     pickle.dump(train_enriched, f, pickle.HIGHEST_PROTOCOL)

# make TimeSinceInset constant and age
for ID in TAMC.ID.unique():
    time_since_onset = TAMC.loc[TAMC['ID'] == ID, 'TimeSinceOnset'].min()
    TAMC.loc[TAMC['ID'] == ID, 'TimeSinceOnset'] = time_since_onset
    birth_year = TAMC.loc[TAMC['ID'] == ID, 'BirthYear'].values[0]
    year = int(TAMC.loc[TAMC['ID'] == ID, 'VisitDate'].values[0].split('-')[0])
    age_ID = year - birth_year
    TAMC.loc[TAMC['ID'] == ID, 'Age'] = age_ID

# ID: 763   Missing: 12,631     Rows without missing: 3,443  ID without missing: 626
TAMC = ts.backward_imputation(TAMC, FRS, only_if_max=4)
# ID: 763   Missing: 12,329     Rows without missing: 3,443  ID without missing: 626
TAMC = ts.backward_imputation(TAMC, MMT, only_if_max=5)
# ID: 763   Missing: 11,887     Rows without missing: 3,457  ID without missing: 626
TAMC = ts.backward_imputation(TAMC, FRS_TOTAL, only_if_max=40)
# ID: 763   Missing: 11,886     Rows without missing: 3,457  ID without missing: 626
TAMC = ts.forward_imputation(TAMC, FRS+MMT+FRS_TOTAL, only_if_min=0)
# ID: 763   Missing: 11,010     Rows without missing: 3,464  ID without missing: 626
TAMC = ts.simple_average_interpolation_imputation(TAMC, FRS+MMT+FRS_TOTAL, only_if_equals=True)
# ID: 763   Missing: 7,782     Rows without missing: 3,510  ID without missing: 626
TAMC['ALSFRS'].isna().sum() # 190 missing in ALSFRS
TAMC[FRS].sum(1, skipna=False).isna().sum() # 161 missing in ALSFRS

TAMC['ALSFRS'] = TAMC[FRS].sum(1, skipna=False)


TAMC.drop(['Education', 'VisitDate', 'OnsetDate', 'BirthYear'], 1, inplace=True)

TAMC.loc[:, 'isSmoking'] = np.nan
TAMC.loc[TAMC['Smoking'] > 0, 'isSmoking'] = 1
TAMC.loc[TAMC['Smoking'] == 0, 'isSmoking'] = -1
TAMC.drop('Smoking', 1, inplace=True)
TAMC.loc[TAMC['Familial'] == 0, 'Familial'] = -1
TAMC.loc[TAMC['Dementia'] == 0, 'Dementia'] = -1
TAMC.loc[TAMC['Sport'] == 0, 'Sport'] = -1
TAMC.loc[TAMC['OnsetSite'] == 0, 'OnsetSite'] = -1
TAMC.loc[TAMC['Gender'] == 0, 'Gender'] = -1

ts.add_missing_indicator(TAMC, 'isSmoking', inplace=True)
ts.add_missing_indicator(TAMC, 'Familial', inplace=True)
ts.add_missing_indicator(TAMC, 'Dementia', inplace=True)
ts.add_missing_indicator(TAMC, 'Sport', inplace=True)
ts.add_missing_indicator(TAMC, 'OnsetSite', inplace=True)
ts.add_missing_indicator(TAMC, 'Gender', inplace=True)

#
# def number_of_patients(data):
#     counter = 0
#     for ID in data.ID.unique():
#         has_three_visits = data[data['ID'] == ID].shape[0] >= 3
#         if has_three_visits:
#             valid_y = not all(np.isnan(data[data['ID'] == ID]['y'].values[2:]))
#             if valid_y:
#                 counter += 1
#     print(counter)


# ----------------------------------------- #
# ------------- Data for exp1 ------------- #
# ----------------------------------------- #
TAMC1 = TAMC.copy()
features = ['AgeOnset', 'Gender', 'Sport', 'OnsetSite', 'Familial',
            'Dementia', 'RArmAbduct', 'RElbowFlex', 'RElbowExt', 'RFingerAbduct',
            'RFingerExt', 'LArmAbduct', 'LElbowFlex', 'LElbowExt', 'LFingerAbduct',
            'LFingerExt', 'RThighFlex', 'RKneeExt', 'RAnkleDorsi', 'RAnklePlantar',
            'LThighFlex', 'LKneeExt', 'LAnkleDorsi', 'LAnklePlantar', 'Speech',
            'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Turning',
            'Walking', 'Stairs', 'Dyspnea', 'ALSFRS', 'time', 'TimeSinceOnset',
            'no', 'Age', 'isSmoking', 'isSmokingNA', 'FamilialNA',
            'DementiaNA', 'SportNA', 'OnsetSiteNA', 'GenderNA']
TAMC1.dropna(subset=features, inplace=True)
TAMC1.sort_values(['ID', 'no', 'time'], inplace=True)
TAMC1.reset_index(drop=True, inplace=True)

valid_patients = []
for ID in TAMC1.ID.unique():
    has_three_visits = TAMC1[TAMC1['ID'] == ID].shape[0] >= 3
    if has_three_visits:
        valid_y = not all(np.isnan(TAMC1[TAMC1['ID'] == ID]['y'].values[2:]))
        if valid_y:
            valid_patients.append(ID)

TAMC1 = TAMC1[TAMC1.ID.isin(valid_patients)]
TAMC1.reset_index(drop=True, inplace=True)
ts.add_ordering(TAMC1, inplace=True)

TAMC1.to_csv(exp1_data_path + 'TAMC1.csv', index=False)


split = dict()
for i in range(60):
    split[i] = dict()
    val_size = test_size = int(0.1 * TAMC1.ID.nunique())
    train_pats, test_pats = train_test_split(TAMC1.ID.unique(), test_size=test_size, random_state=i)
    split[i]['train_pats'] = train_pats
    split[i]['test_pats'] = test_pats


with open(exp1_data_path + 'TAMC_1_train_test_split.pkl', 'wb') as f:
    pickle.dump(split, f, pickle.HIGHEST_PROTOCOL)

# ----------------------------------------- #
# ------------- Data for exp2 ------------- #
# ----------------------------------------- #
TAMC2 = TAMC.copy()

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features

TAMC2['Onset.Limb'] = TAMC2['OnsetSite'].apply(lambda x: 1 if x == -1 else -1)
TAMC2['Onset.Bulbar'] = TAMC2['OnsetSite'].apply(lambda x: 1 if x == 1 else -1)

TAMC2 = TAMC2[['ID', 'time', 'no'] + features + ['y']]

TAMC2.dropna(subset=features, inplace=True)
TAMC2.sort_values(['ID', 'no', 'time'], inplace=True)
TAMC2.reset_index(drop=True, inplace=True)
ts.add_ordering(TAMC2, inplace=True)

valid_patients = []
for ID in TAMC2.ID.unique():
    has_three_visits = TAMC2[TAMC2['ID'] == ID].shape[0] >= 3
    if has_three_visits:
        valid_y = not all(np.isnan(TAMC2[TAMC2['ID'] == ID]['y'].values[2:]))
        if valid_y:
            valid_patients.append(ID)

TAMC2 = TAMC2[TAMC2.ID.isin(valid_patients)]
TAMC2.reset_index(drop=True, inplace=True)

TAMC2.to_csv(exp2_path + 'TAMC2.csv', index=False)


split = dict()
for i in range(60):
    split[i] = dict()
    val_size = test_size = int(0.1 * TAMC2.ID.nunique())
    train_pats, test_pats = train_test_split(TAMC2.ID.unique(), test_size=test_size, random_state=i)
    split[i]['train_pats'] = train_pats
    split[i]['test_pats'] = test_pats

with open(exp2_path + 'TAMC_2_train_test_split.pkl', 'wb') as f:
    pickle.dump(split, f, pickle.HIGHEST_PROTOCOL)
