"""
Input: flattened data
processes:
    b) down-casting

"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression


def drop_correlated(x, y, thresh=0.9):
    x_cur = x.copy()
    features_dropped = []
    STOP = False
    counter = 0
    while not STOP:
        cor_mat = x_cur.corr().abs()
        upper = cor_mat.where(np.triu(np.ones(cor_mat.shape), k=1).astype(np.bool))
        max_correlation = upper.max().max()
        i, j = np.argwhere(upper.values == max_correlation)[0]

        c1 = upper.columns[i]
        c2 = upper.columns[j]
        print(f"{counter} ))\t Candidates: {c1} and {c2} with corr: {max_correlation:.2f}")

        if max_correlation == 1:
            print(f"drop {c1}")
            x_cur.drop(c1, 1, inplace=True)
            features_dropped.append(c1)
        elif max_correlation >= thresh:
            X_tmp = x[[c1]]
            f_score_1 = f_regression(X_tmp[~X_tmp[c1].isna()], y[~X_tmp[c1].isna()])[0]
            X_tmp = x[[c2]]
            f_score_2 = f_regression(X_tmp[~X_tmp[c2].isna()], y[~X_tmp[c2].isna()])[0]
            if f_score_1 > f_score_2:
                print(f"drop {c2}")
                x_cur.drop(c2, 1, inplace=True)
                features_dropped.append(c1)
            else:
                x_cur.drop(c1, 1, inplace=True)
                print(f"drop {c1}")
                features_dropped.append(c1)
        else:
            STOP = True
        counter += 1
    return features_dropped


def drop_with_missing_values(x, thresh):
    missing = x.isna().sum().sort_values(ascending=False) / x.shape[0]
    features = missing[missing > thresh].index.tolist()
    return features


def drop_ID_temp_1_if_exist(X):
    if 'ID_temp.1' in X.columns:
        X.drop(['ID_temp.1'], 1, inplace=True)


def drop_kurt_if_exist(X):
    kurt = [col for col in X.columns if col.split('_')[-1] == 'kurt']  # TODO: check permutation number 1
    if len(kurt) > 0:
        X.drop(kurt, 1, inplace=True)


def downcast_x(X):
    for column in X.columns:
        if X[column].dtype != 'O':
            X[column] = pd.to_numeric(X[column], downcast='integer')
            X[column] = pd.to_numeric(X[column], downcast='float')
            if X[column].min() >= 0:
                X[column] = pd.to_numeric(X[column], downcast='unsigned')
    return X


def main(data_name):
    X = None
    if data_name == 'tasmc':
        X = pd.read_csv('f14_tasmc_flat.csv')
    elif data_name == 'proact':
        X = pd.read_csv('f04_proact_flat.csv') if exp == 1 else pd.read_csv('Exp2/f04_proact_flat.csv')

    X = X.replace([-np.inf, np.inf], np.nan)  # replace -inf with nan
    y = X['y']
    X = X.drop('y', 1)

    features_to_drop_1 = []
    features_to_drop_2 = drop_with_missing_values(X, 0.4)
    print(features_to_drop_2)
    X.drop(features_to_drop_2, 1, inplace=True)

    print("Save...", end=' ')
    if data_name == 'tasmc':
        f = open(f'Exp{exp}/f18_features_to_drop_from_tasmc_flat.txt', 'w')
        print("TASMC saved!")
    elif data_name == 'proact':
        f = open(f'Exp{exp}/f19_features_to_drop_from_proact_flat.txt', 'w')
        print("PROACT saved!")

    for feature in features_to_drop_1 + features_to_drop_2:
        f.write(feature + '\n')
    f.close()

    drop_kurt_if_exist(X)
    drop_ID_temp_1_if_exist(X)
    X = downcast_x(X)

    X['y'] = y

    new_data_file = ''
    if data_name == 'tasmc':
        new_data_file = f'Exp{exp}/data/tasmc_static/tasmc_flat.csv'
    elif data_name == 'proact':
        new_data_file = f'Exp{exp}/data/proact_static/proact_flat.csv'

    X.to_csv(new_data_file, index=False)


if __name__ == '__main__':
    exp = 1
    main('tasmc')
    # main('proact')
