import pandas as pd
import numpy as np
import TSFunctions as ts

folder = 'C:/Users/Ben/Desktop/Results/'
proact_file = 'C:/Users/Ben/Desktop/Data/PROACT/data.csv'
proact = pd.read_csv(proact_file)


# Exp2
save_to = folder + 'PROACT2.csv'
proact['Onset.time'] *= (-1)

proact['y'] = proact['ALSFRS_Total']

proact.rename(columns={'Q1_Speech': 'Speech',
                       'Q2_Salivation': 'Salivation',
                       'Q3_Swallowing': 'Swallowing',
                       'Q4_Handwriting': 'Writing',
                       'Q5_Cutting': 'Food',
                       'Q6_Dressing_and_Hygiene': 'Dressing',
                       'Q7_Tutning_in_Bed': 'Turning',
                       'Q8_Walking': 'Walking',
                       'Q9_Climbing_Stairs': 'Stairs',
                       'Q10_Respiratory': 'Dyspnea',
                       'Onset.time': 'TimeSinceOnset',
                       'ALSFRS_Total': 'ALSFRS',
                       'Sex.Male': 'Gender'},
              inplace=True)

# Drop patients with less than 3 visits
pats_to_stay = []
counter = 0
for ID in proact.ID.unique():
    if proact[proact['ID'] == ID].shape[0] < 3:
        counter += 1
    else:
        pats_to_stay.append(ID)
print(f'{counter} patients dropped')

proact = proact[proact.ID.isin(pats_to_stay)]

FRS = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea']
temporal_features = ['Speech', 'Salivation', 'Swallowing', 'Writing', 'Food', 'Dressing', 'Walking', 'Stairs', 'Dyspnea', 'ALSFRS']
constant_features = ['Age', 'TimeSinceOnset', 'Onset.Bulbar', 'Onset.Limb', 'Gender']
features = temporal_features + constant_features
proact = proact[['ID', 'time'] + features + ['y']]

ts.add_ordering(proact, inplace=True)
proact.to_csv(save_to, index=False)