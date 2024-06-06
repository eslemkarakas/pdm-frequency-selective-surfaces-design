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
    MODEL_SAVE_PATH = os.path.join(CUR_PATH, 'src/models/')
    RADIUS_COLS = ['r0', 'r1', 'r2', 'r3', 'r4']
    BASE_COLS = ['f1', 'f2']
    TEST_SIZE = 0.1
    
def train_model():
    df = pd.read_csv(C.EDITED_DF_PATH)
    
    for i in range(len(C.RADIUS_COLS)):
        X, y = call_subchunker(df, C.BASE_COLS, C.RADIUS_COLS, scope_idx=i)
        optimizer = call_optimizer(X, y, metric='mae')
        best_params = optimizer.get_best_params()
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=42, shuffle=True)
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        xgb_model = xgb.train(best_params, dtrain, 100, evals=[(dvalid, 'test')], early_stopping_rounds=10)

        model_path = os.path.join(C.MODEL_SAVE_PATH, f"xgb_model_{i}.json")
        xgb_model.save_model(model_path)
        print(f"Model for iteration {i} saved at {model_path}")
        
        y_pred = xgb_model.predict(dvalid)
        mae = mean_absolute_error(y_valid, y_pred)
        print(f"{i}th Iteration - Mean Absolute Error:", mae)

def load_and_predict(f1, f2):
    r_outputs = [] # store model outputs
    X = np.array([f1, f2]).reshape(1, -1) # start with first input
    
    for model_id in range(5): # model should run 5 times for each r value from r1 to r5
        model_path = os.path.join(C.MODEL_SAVE_PATH, f"xgb_model_{model_id}.json")
        model = xgb.Booster()
        model.load_model(model_path)
    
        ddata = xgb.DMatrix(X)
        pred = model.predict(ddata)
        r_outputs.append(float(pred))
        X = np.array([f1, f2].extend(r_outputs)).reshape(1, -1) # update input to apply expanding model approach

    r_outputs.sort() # sort not to face with logical error between r values (ex. r0>r4)
    
    return r_outputs