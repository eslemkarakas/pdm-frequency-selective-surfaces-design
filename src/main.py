# import standard packages
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import local packages
from utils.subchunker import call_subchunker

# suppress warnings
warnings.filterwarnings('ignore')

# define static configurations
class C:
    CUR_PATH = os.getcwd()
    EDITED_DF_PATH = os.path.join(CUR_PATH, 'src/data/freq_surface.csv')
    RADIUS_COLS = ['r0', 'r1', 'r2', 'r3', 'r4']
    BASE_COLS = ['f1', 'f2']
    TEST_SIZE = 0.1
    
# load data
df = pd.read_csv(C.EDITED_DF_PATH)
 
# define parameters
params = {
          'eval_metric': 'error',
          'max_depth': 6,
          'learning_rate': 0.1,
          'seed': 42
         }

for i in range(len(C.RADIUS_COLS)):
    # create train and test set in order of iteration
    X, y = call_subchunker(df, C.BASE_COLS, C.RADIUS_COLS, scope_idx=i)

    # split dataframe as train and validation sets 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=42, shuffle=True)

    # create xgboost-specified dmatrix for train set
    dtrain = xgb.DMatrix(data=X_train, label=y_train)

    # create xgboost-specified dmatrix for validation set
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

    # train the model
    num_rounds = 100
    xgb_model = xgb.train(params, dtrain, num_rounds, evals=[(dvalid, 'test')], early_stopping_rounds=10)

    # evaluate the model
    y_pred = xgb_model.predict(dvalid)
    mae = mean_absolute_error(y_valid, y_pred)
    print(f"{i}th Iteration - Mean Absolute Error:", mae)