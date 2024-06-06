# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.subchunker import call_subchunker
from optimizer.optuna import call_optimizer

warnings.filterwarnings('ignore')

class C:
    CUR_PATH = os.getcwd()
    EDITED_DF_PATH = os.path.join(CUR_PATH, 'src/data/freq_surface.csv')
    RADIUS_COLS = ['r0', 'r1', 'r2', 'r3', 'r4']
    BASE_COLS = ['f1', 'f2']
    TEST_SIZE = 0.1
    
df = pd.read_csv(C.EDITED_DF_PATH)
 
for i in range(len(C.RADIUS_COLS)):
    X, y = call_subchunker(df, C.BASE_COLS, C.RADIUS_COLS, scope_idx=i)
    optimizer = call_optimizer(X, y, metric='mae')
    best_params = optimizer.get_best_params()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=42, shuffle=True)
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
    xgb_model = xgb.train(best_params, dtrain, 100, evals=[(dvalid, 'test')], early_stopping_rounds=10)

    y_pred = xgb_model.predict(dvalid)
    mae = mean_absolute_error(y_valid, y_pred)
    print(f"{i}th Iteration - Mean Absolute Error:", mae)