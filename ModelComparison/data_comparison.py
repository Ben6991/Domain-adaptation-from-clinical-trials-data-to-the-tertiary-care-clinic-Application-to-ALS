# Train model based on PROACT, test model on TAMC database
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prompt_toolkit.styles import named_colors
from tqdm import tqdm

plots_folder = 'ModelComparison/plots/'
proact_file = 'C:/Users/Ben/Desktop/Data/PROACT/data.csv'
tamc_file = 'C:/Users/Ben/Desktop/072019.csv'

PROACT = pd.read_csv(proact_file)
TAMC = pd.read_csv(tamc_file)


features = ['ID', 'time', 'Age', 'TimeSinceOnset', 'Onset.Bulbar',
            'Onset.Limb', 'Gender', 'Speech', 'Salivation','Swallowing',
            'Writing', 'Food', 'Dressing', 'Walking', 'Stairs',
            'Resp', 'ALSFRS']

proact = proact[features]
tamcs = tamc[features]

proact.reset_index(drop=True, inplace=True)
tamc.reset_index(drop=True, inplace=True)

# Drop FVC (high amount of missing values
# test_data.drop('FVC', 1, inplace=True)
# train_data.drop('FVC', 1, inplace=True)

# Drop rows contain missing values
# train_data.dropna(inplace=True)
# test_data.dropna(inplace=True)

# Drop patients with less than 3 visits
import TSFunctions as ts
ts.drop_by_number_of_records(tamc, 3, inplace=True, verbose=2)
ts.drop_by_number_of_records(proact, 3, inplace=True, verbose=2)

proact.reset_index(drop=True, inplace=True)
tamc.reset_index(drop=True, inplace=True)

data = dict(PROACT=proact, TAMC=tamc)
import pickle
with open('C:/Users/Ben/Desktop/Results/data_proact_tamc.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

for ID in TAMC.ID.unique():
    time_since_onset = TAMC.loc[TAMC['ID'] == ID, 'TimeSinceOnset'].min()
    TAMC.loc[TAMC['ID'] == ID, 'TimeSinceOnset'] = time_since_onset
    birth_year = TAMC.loc[TAMC['ID'] == ID, 'BirthYear'].values[0]
    year = int(TAMC.loc[TAMC['ID'] == ID, 'VisitDate'].values[0].split('-')[0])
    age_ID = year - birth_year
    TAMC.loc[TAMC['ID'] == ID, 'Age'] = age_ID

# Convert OnsetSite to Onset.Limb and Onset.Bulbar
TAMC['Onset.Limb'] = TAMC['OnsetSite'].apply(lambda x: 1 if x == 0 else -1)
TAMC['Onset.Bulbar'] = TAMC['OnsetSite'].apply(lambda x: 1 if x == 1 else -1)

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# --------------- Visualisation of databases comparison -------------- #
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

# Range of time
proact_t_range = PROACT[['ID', 'time']].groupby('ID').max().reset_index()
proact_t_range['database'] = 'PROACT'
tamc_t_range = TAMC[['ID', 'time']].groupby('ID').max().reset_index()
tamc_t_range['database'] = 'TAMC'
time_range = pd.concat([proact_t_range, tamc_t_range])

# Time range
sns.boxplot(x='database', y='time', data=time_range)
plt.xlabel("Database", fontsize=15)
plt.ylabel("Days", fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.axes().set_axisbelow(True)
plt.grid(axis='y', alpha=0.7)
plt.savefig(plots_folder + 'comp_time_range.png', bbox_inches='tight')
plt.show()

# Age
a = PROACT[['ID', 'Age']].drop_duplicates()[['Age']]
a['database'] = 'PROACT'
b = TAMC[['ID', 'Age']].drop_duplicates()[['Age']]
b['database'] = 'TAMC'
age = pd.concat([a, b])

sns.boxplot(x='database', y='Age', data=age)
plt.title("Age comparison")
plt.xlabel("Database", fontsize=15)
plt.ylabel("Age", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axes().set_axisbelow(True)
plt.grid(axis='y', alpha=0.7)
plt.show()

# Onset site
a = PROACT[['ID', 'Onset.Limb']].drop_duplicates()[['Onset.Limb']]
a['Onset.Limb'] = a['Onset.Limb'].apply(lambda x: 'Limb' if x == 1 else 'Bulbar')
b = TAMC[['ID', 'Onset.Limb']].drop_duplicates()[['Onset.Limb']]
b['Onset.Limb'] = b['Onset.Limb'].apply(lambda x: 'Limb' if x == 1 else 'Bulbar')

a = PROACT[['ID', 'Onset.Limb', 'Onset.Bulbar']].drop_duplicates()
b = TAMC[['ID', 'Onset.Limb', 'Onset.Bulbar']].drop_duplicates()

a.loc[(a['Onset.Limb'] == 1) & (a['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Limb'
a.loc[(a['Onset.Limb'] == -1) & (a['Onset.Bulbar'] == 1), 'OnsetSite'] = 'Bulbar'
a.loc[(a['Onset.Limb'] == -1) & (a['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Unknown'
b.loc[(b['Onset.Limb'] == 1) & (b['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Limb'
b.loc[(b['Onset.Limb'] == -1) & (b['Onset.Bulbar'] == 1), 'OnsetSite'] = 'Bulbar'
b.loc[(b['Onset.Limb'] == -1) & (b['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Unknown'
a['database'] = 'PROACT'
b['database'] = 'TAMC'
onset_site = pd.concat([a, b])


fig = sns.countplot(x='database', hue='OnsetSite', data=onset_site)
plt.title("Onset site comparison")
plt.xlabel("Database", fontsize=15)
plt.ylabel("Onset site", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Onset site')
plt.axes().set_axisbelow(True)
plt.ylim(0, 2620)
for p in fig.patches:
    fig.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
plt.grid(axis='y', alpha=0.7)
plt.show()

# Gender
PROACT['Gender'] = PROACT['Sex.Male'].apply(lambda x: -1 if x == -1 else 1)
TAMC['Gender'] = TAMC['Gender'].apply(lambda x: -1 if x == 0 else 1)

a = PROACT[['ID', 'Gender']].drop_duplicates()
a['database'] = 'PROACT'
b = TAMC[['ID', 'Gender']].drop_duplicates()
b['database'] = 'TAMC'
gender=pd.concat([a, b])
gender['Gender'] = gender['Gender'].apply(lambda x: 'Male' if x==1 else 'Female')

fig = sns.countplot(x='database', hue='Gender', data=gender)
plt.xlabel("Database", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 2200)
plt.axes().set_axisbelow(True)
for p in fig.patches:
    fig.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
plt.grid(axis='y', alpha=0.7)
plt.show()


# TimeSinceOnset
PROACT['TimeSinceOnset'] = PROACT['Onset.time']*(-1)
a = PROACT[['ID', 'TimeSinceOnset']].drop_duplicates()
a['database'] = 'PROACT'
b = TAMC[['ID', 'TimeSinceOnset']].drop_duplicates()
b['database'] = 'TAMC'
time_since_onset = pd.concat([a, b])

sns.boxplot(x='database', y='TimeSinceOnset', data=time_since_onset)
plt.title("Time since onset comparison")
plt.xlabel("Database")
plt.ylabel("Days")
plt.axes().set_axisbelow(True)
plt.grid(axis='y', alpha=0.7)
plt.show()

# Number of visits
a = train_data[['ID']].groupby('ID').size().reset_index(drop=True)
b = test_data[['ID']].groupby('ID').size().reset_index(drop=True)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
sns.countplot(a, ax=axes[0], color='teal')
sns.countplot(b, ax=axes[1], color='teal')
axes[0].set_title('PROACT')
axes[1].set_title('TAMC')
axes[0].set_ylabel('Frequency')
axes[1].set_ylabel('Frequency')
axes[1].set_xlabel("Number of visits")
plt.show()

# time between visits
ts.add_ordering(train_data, inplace=True)
ts.add_ordering(test_data, inplace=True)

train_data.sort_values(['ID', 'no'], inplace=True)
test_data.sort_values(['ID', 'no'], inplace=True)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)


def create_time_between(data):
    for i in tqdm(range(data.shape[0])):
        if i < data.shape[0]-1:
            next_visit = data.loc[i+1, 'no']
            if next_visit == 1:
                data.loc[i, 'TimeBet'] = np.nan
            else:
                t_cur = data.loc[i, 'time']
                t_next = data.loc[i+1, 'time']
                data.loc[i, 'TimeBet'] = t_next - t_cur
                if t_next - t_cur < 0:
                    print("ERROR")
        else:
            data.loc[i, 'TimeBet'] = np.nan


create_time_between(train_data)
create_time_between(test_data)

a = train_data[['TimeBet']].copy()
b = test_data[['TimeBet']].copy()
a['database'] = 'PROACT'
b['database'] = 'TAMC'
time_between = pd.concat([a.dropna(), b.dropna()])

time_between[time_between.database == 'PROACT']['TimeBet'].std()
time_between[time_between.database == 'TAMC']['TimeBet'].std()

sns.boxplot(x='database', y='TimeBet', data=time_between)
plt.title("Time between visits comparison")
plt.ylabel("Time between visits (days)")
plt.xlabel("Database")
plt.show()


print('\t\tTime between visits')
print(f'PROACT:\t\t\t{a.TimeBet.mean():.2f} ({a.TimeBet.std():.2f})')
print(f'TAMC\t\t\t{b.TimeBet.mean():.2f} ({b.TimeBet.std():.2f})')

# ------------------------------ #
# ------  All information ------ #
# ------------------------------ #

# --- Time since onset --- #
feature = 'TimeSinceOnset'
proact_feat = PROACT[['ID', 'TimeSinceOnset']].drop_duplicates()['TimeSinceOnset'] / (365/12) / 12
tamc_feat = TAMC[['ID', 'TimeSinceOnset']].drop_duplicates()['TimeSinceOnset'] / (365/12) / 12
print(f'\t:: PROACT :: \nMin: {proact_feat.min():.3f}\nMax: {proact_feat.max():.3f}\nMean: {proact_feat.mean():.3f}\nstd: {proact_feat.std():.3f}')
print(f'\t:: TAMC :: \nMin: {tamc_feat.min():.3f}\nMax: {tamc_feat.max():.3f}\nMean: {tamc_feat.mean():.3f}\nstd: {tamc_feat.std():.3f}')

# --- Number of visits --- #
feature = 'Number of visits'
a = PROACT[['ID']].groupby('ID').size().reset_index()
a.columns = ['ID', 'visits']
b = TAMC[['ID']].groupby('ID').size().reset_index()
b.columns = ['ID', 'visits']

proact_feat = a.visits
tamc_feat = b.visits
print(f'\t:: PROACT :: \nMin: {proact_feat.min():.3f}\nMax: {proact_feat.max():.3f}\nMean: {proact_feat.mean():.3f}\nstd: {proact_feat.std():.3f}')
print(f'\t:: TAMC :: \nMin: {tamc_feat.min():.3f}\nMax: {tamc_feat.max():.3f}\nMean: {tamc_feat.mean():.3f}\nstd: {tamc_feat.std():.3f}')


# --- Age --- #
feature = 'Age'
proact_feat = PROACT[['ID', 'Age']].groupby('ID').min().reset_index().Age
tamc_feat = TAMC[['ID', 'Age']].groupby('ID').min().reset_index().Age

print(f'\t:: PROACT :: \nMin: {proact_feat.min():.3f}\nMax: {proact_feat.max():.3f}\nMean: {proact_feat.mean():.3f}\nstd: {proact_feat.std():.3f}')
print(f'\t:: TAMC :: \nMin: {tamc_feat.min():.3f}\nMax: {tamc_feat.max():.3f}\nMean: {tamc_feat.mean():.3f}\nstd: {tamc_feat.std():.3f}')


# --- Time between visits --- #
import TSFunctions as ts

tamc_time_between = []
for ID in TAMC.ID.unique():
    for visit in TAMC[TAMC['ID'] == ID].no.values[:-1]:
        time_bet = ts.time_between_records(TAMC, visit, visit+1, ID)
        if time_bet != 0:
            tamc_time_between.append(time_bet)


ts.add_ordering(PROACT)
proact_time_between = []
for ID in PROACT.ID.unique():
    for visit in PROACT[PROACT['ID'] == ID].no.values[:-1]:
        time_bet = ts.time_between_records(PROACT, visit, visit+1, ID)
        if time_bet != 0:
            proact_time_between.append(time_bet)

proact_feat = np.array(proact_time_between) / (365/12)
tamc_feat = np.array(tamc_time_between) / (365/12)

print(f'\t:: PROACT :: \nMin: {proact_feat.min():.3f}\nMax: {proact_feat.max():.3f}\nMean: {proact_feat.mean():.3f}\nstd: {proact_feat.std():.3f}')
print(f'\t:: TAMC :: \nMin: {tamc_feat.min():.3f}\nMax: {tamc_feat.max():.3f}\nMean: {tamc_feat.mean():.3f}\nstd: {tamc_feat.std():.3f}')


# Time range
proact_t_min = PROACT[['ID', 'time']].groupby('ID').min()
proact_t_max = PROACT[['ID', 'time']].groupby('ID').max()
proact_feat = ((proact_t_max - proact_t_min) / (365/12)).values
tamc_feat = (TAMC[['ID', 'time']].groupby('ID').max().time / (365/12)).values
print(f'\t:: PROACT :: \nMin: {proact_feat.min():.3f}\nMax: {proact_feat.max():.3f}\nMean: {proact_feat.mean():.3f}\nstd: {proact_feat.std():.3f}')
print(f'\t:: TAMC :: \nMin: {tamc_feat.min():.3f}\nMax: {tamc_feat.max():.3f}\nMean: {tamc_feat.mean():.3f}\nstd: {tamc_feat.std():.3f}')


# Gender
PROACT[['ID', 'Gender']].drop_duplicates()['Gender'].value_counts()*100 / PROACT.ID.nunique()
TAMC[['ID', 'Gender']].drop_duplicates()['Gender'].value_counts()*100/ TAMC.ID.nunique()

# Onset site
a = PROACT[['ID', 'Onset.Limb']].drop_duplicates()[['Onset.Limb']]
a['Onset.Limb'] = a['Onset.Limb'].apply(lambda x: 'Limb' if x == 1 else 'Bulbar')
b = TAMC[['ID', 'Onset.Limb']].drop_duplicates()[['Onset.Limb']]
b['Onset.Limb'] = b['Onset.Limb'].apply(lambda x: 'Limb' if x == 1 else 'Bulbar')

a = PROACT[['ID', 'Onset.Limb', 'Onset.Bulbar']].drop_duplicates()
b = TAMC[['ID', 'Onset.Limb', 'Onset.Bulbar']].drop_duplicates()


100-58.36
a.loc[(a['Onset.Limb'] == 1) & (a['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Limb'
a.loc[(a['Onset.Limb'] == -1) & (a['Onset.Bulbar'] == 1), 'OnsetSite'] = 'Bulbar'
a.loc[(a['Onset.Limb'] == -1) & (a['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Unknown'
b.loc[(b['Onset.Limb'] == 1) & (b['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Limb'
b.loc[(b['Onset.Limb'] == -1) & (b['Onset.Bulbar'] == 1), 'OnsetSite'] = 'Bulbar'
b.loc[(b['Onset.Limb'] == -1) & (b['Onset.Bulbar'] == -1), 'OnsetSite'] = 'Unknown'
a['database'] = 'PROACT'
b['database'] = 'TAMC'
onset_site = pd.concat([a, b])

onset_site[onset_site.database == 'PROACT']['OnsetSite'].value_counts() / PROACT.ID.nunique()
onset_site[onset_site.database == 'TAMC']['OnsetSite'].value_counts() / TAMC.ID.nunique()

