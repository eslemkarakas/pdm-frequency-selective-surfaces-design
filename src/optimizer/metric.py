# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def mse(y_true, y_pred):
    result = mean_squared_error(y_true, y_pred)
    return result
    
def mae(y_true, y_pred):
    result = mean_absolute_error(y_true, y_pred)
    return result
    
def mape(y_true, y_pred):
    result = mean_absolute_percentage_error(y_true, y_pred)
    return result
    
def rmse(y_true, y_pred):
    result = np.sqrt(mean_squared_error(y_true, y_pred))
    return result

class Metrics:
    def __init__(self, metric):
        self.metric = metric

        if self.metric == 'mae':
            self.metric = mae
        elif self.metric == 'rmse':
            self.metric = rmse
        elif self.metric == 'mape':
            self.metric = mape
        else:
            pass
            
    def __call__(self, y_true, y_pred):
        return self.metric(y_true, y_pred)
    
def call_metric(metric):
    method = None
    try:
        method = Metrics(metric)
    except Exception as e:
        print(f'ERROR: Could not call optimizer class as right - {str(e)}')
        
    return method