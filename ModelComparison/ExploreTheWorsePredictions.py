import numpy as np
import pandas as pd

a = pd.read_csv('C:/Users/Ben/Desktop/Results/train_PROACT_test_TAMC/XGB/predictions/0.csv')

(a['y_true'] - a['pred']).abs()