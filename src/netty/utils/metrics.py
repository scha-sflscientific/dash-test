import pandas as pd
import numpy as np
import sys
import logging
from tqdm import tqdm
import pdb


def mean_absolute_percentage_error(y_true, y_pred):
    y_true_temp = []
    y_pred_temp = []
    for x, y in zip(y_true, y_pred):
        if x == 0 and y == 0:
            y_true_temp.append(1)
            y_pred_temp.append(1)
        elif x != 0:
            y_true_temp.append(x)
            y_pred_temp.append(y)

    y_true, y_pred = np.array(y_true_temp), np.array(y_pred_temp)
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100
