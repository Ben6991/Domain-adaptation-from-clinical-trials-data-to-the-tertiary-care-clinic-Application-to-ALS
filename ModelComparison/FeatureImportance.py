import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
import TSFunctions as ts
from xgboost import XGBRegressor
import seaborn as sns

folder = 'C:/Users/Ben/Desktop/Results/'

# ------------------------------------------ #
# ------------------- RF ------------------- #
# ------------------------------------------ #

#
# ------------------- TAMC ------------------- #
#

res = pd.DataFrame()
for i in range(60):
    fi = pd.read_csv(folder + f'TAMC/RF/feature importances/{i}.csv')
    fi['iter'] = i
    res = pd.concat([res, fi])

# mean importance
res_mean = res.groupby('feature').mean().sort_values('importance', ascending=False).reset_index()

plt.figure(figsize=(5, 3))
sns.barplot(x='feature', y='importance', data=res_mean.head(20), edgecolor='k')
plt.grid(axis='y')
plt.xlabel("Feature", fontsize=15)
plt.ylabel("Importance", fontsize=15)
plt.xticks(rotation=90)
plt.axes().set_axisbelow(True)
plt.savefig('C:/Users/Ben/PycharmProjects/master/ModelComparison/plots/RF_feature_importance_top_20.png', bbox_inches='tight')
plt.show()

#
# ------------------- PROACT ------------------- #
#


res = pd.read_csv(folder + 'PROACT_RF_importance.csv')


# mean importance
res_mean = res.groupby('feature').mean().sort_values('importance', ascending=False).reset_index()

plt.figure(figsize=(5, 3))
sns.barplot(x='feature', y='importance', data=res_mean.head(20), edgecolor='k')
plt.grid(axis='y')
plt.xlabel("Feature", fontsize=15)
plt.ylabel("Importance", fontsize=15)
plt.xticks(rotation=90)
plt.axes().set_axisbelow(True)
plt.savefig('C:/Users/Ben/PycharmProjects/master/ModelComparison/plots/RF_PROACT_feature_importance_top_20.png', bbox_inches='tight')
plt.show()












# ----------------------------------------------------------- #
# ------------------------- TAMC RF ------------------------- #
# ----------------------------------------------------------- #

feature_importance = pd.DataFrame(columns=['feature', 'importance', 'iter'])
model_name = 'RF'
for i in range(60):
    temp = pd.read_csv(f'C:/Users/Ben/Desktop/Results/new_TAMC/RF/feature importances/{i}.csv')
    temp['iter'] = i
    feature_importance = pd.concat([feature_importance, temp])


aa = feature_importance.groupby('feature').max().reset_index().sort_values('importance', ascending=False).head(30)
plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig('ModelComparison/plots/TASMC_RF_feature_imprtance_top_15.png', quality=100, bbox_inches='tight')
plt.show()


aa = feature_importance.groupby('feature').max().reset_index().sort_values('importance', ascending=False).tail(30)
plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.ylim(0, 0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig('ModelComparison/plots/TASMC_RF_feature_imprtance_last_15.png', quality=100, bbox_inches='tight')
plt.show()


# ------------------------------------------------------------ #
# ------------------------- TAMC XGB ------------------------- #
# ------------------------------------------------------------ #

feature_importance = pd.DataFrame(columns=['feature', 'importance', 'iter'])

model_name = 'XGB'
for i in range(60):
    temp = pd.read_csv(f'C:/Users/Ben/Desktop/Results/new_TAMC/{model_name}/feature importances/{i}.csv')
    temp['iter'] = i
    feature_importance = pd.concat([feature_importance, temp])


aa = feature_importance.groupby('feature').max().reset_index().sort_values('importance', ascending=False).head(30)
plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig(f'ModelComparison/plots/TASMC_{model_name}_feature_imprtance_top_30.png', quality=100, bbox_inches='tight')
plt.show()


aa = feature_importance.groupby('feature').max().reset_index().sort_values('importance', ascending=False).tail(30)
plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.ylim(0, 0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig(f'ModelComparison/plots/TASMC_{model_name}_feature_imprtance_last_30.png', quality=100, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------ #
# ------------------------- PROACT RF------------------------- #
# ------------------------------------------------------------ #

with open('C:/Users/Ben/Desktop/Results/XGB_columns.pkl', 'rb') as f:
    columns = pickle.load(f)

PROACT_feature_importances = pd.DataFrame(columns=['feature', 'importance', 'iter'])
for i in range(60):
    try:
        with open(f'C:/Users/Ben/Desktop/Results/PROACT/RF/models/model_{i}.pkl', 'rb') as f:
            a = pickle.load(f)
        temp = pd.DataFrame(dict(feature=columns, importance=a.feature_importances_, iter=i))
        PROACT_feature_importances = pd.concat([PROACT_feature_importances, temp])
    except:
        pass

aa = PROACT_feature_importances.groupby('feature').max().reset_index().sort_values('importance', ascending=False).head(30)

plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig(f'ModelComparison/plots/PROACT_{model_name}_feature_imprtance_top_30.png', quality=100, bbox_inches='tight')
plt.show()

aa = PROACT_feature_importances.groupby('feature').max().reset_index().sort_values('importance', ascending=False).tail(30)
plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.ylim(0, 0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig(f'ModelComparison/plots/PROACT_{model_name}_feature_imprtance_last_30.png', quality=100, bbox_inches='tight')
plt.show()


i = 1
model = XGBRegressor()
model.load_model(f'C:/Users/Ben/Desktop/Results/PROACT/XGB/models/model_{i}.model')

model.feature_importances_

PROACT_feature_importances = pd.DataFrame(columns=['feature', 'importance', 'iter'])
for i in range(60):
    try:
        temp = pd.DataFrame(dict(feature=columns, importance=a.feature_importances_, iter=i))
        PROACT_feature_importances = pd.concat([PROACT_feature_importances, temp])
    except:
        pass

aa = PROACT_feature_importances.groupby('feature').max().reset_index().sort_values('importance', ascending=False).head(30)

plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig(f'ModelComparison/plots/PROACT_{model_name}_feature_imprtance_top_30.png', quality=100, bbox_inches='tight')
plt.show()

aa = PROACT_feature_importances.groupby('feature').max().reset_index().sort_values('importance', ascending=False).tail(30)
plt.figure(figsize=(8, 5))
sns.barplot(x='feature', y='importance', data=aa, edgecolor='k')
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.6)
plt.ylim(0, 0.6)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('Importance', fontsize=20)
plt.axes().set_axisbelow(True)
plt.savefig(f'ModelComparison/plots/PROACT_{model_name}_feature_imprtance_last_30.png', quality=100, bbox_inches='tight')
plt.show()