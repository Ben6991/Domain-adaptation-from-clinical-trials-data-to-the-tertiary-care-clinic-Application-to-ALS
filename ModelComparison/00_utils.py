"""
after creating all static tables, reduce size by down-casting the numeric variables
"""


import pickle

import pandas as pd
from tqdm import tqdm


def downcast_x(X):
    for column in X.columns:
        if X[column].dtype != 'O':
            X[column] = pd.to_numeric(X[column], downcast='integer')
            X[column] = pd.to_numeric(X[column], downcast='float')
            if X[column].min() >= 0:
                X[column] = pd.to_numeric(X[column], downcast='unsigned')
    return X


def downcast_y(y):
    return y.astype('uint8')


