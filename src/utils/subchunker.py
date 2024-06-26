# -*- coding: utf-8 -*-
from tqdm import tqdm

# this function updates X and y variables iterateviley by iterating scope index one-by-one for expanding modeling approach
def find_training_cols(base_columns, radius_columns, scope_idx):
    base_columns.extend(radius_columns[scope_idx-1:scope_idx])
    target_column = radius_columns[scope_idx:scope_idx+1]
    return base_columns, target_column

def create_subchunks(df, train_columns, target_column):
    X, y = [], []
    
    for i in tqdm(range(len(df))):
        X_unit = []
        for train_column in train_columns:
            X_unit.append(float(df[i:i+1][train_column].values))
        X.append(X_unit)
        
        y_unit = [float(df[i:i+1][target_column].values)]
        y.append(y_unit)
        
    return X, y

def call_subchunker(df, base_columns, radius_columns, scope_idx):
    X, y = None, None
    try:
        train_columns, target_column = find_training_cols(base_columns, radius_columns, scope_idx)
        X, y = create_subchunks(df, train_columns, target_column)
    except Exception as e:
        print(f'ERROR: Could not create subchunks - {str(e)}')
        
    return X, y