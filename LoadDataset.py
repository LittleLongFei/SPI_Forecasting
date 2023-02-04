

# 2023-2-3 written by H.Zhang.


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def LoadData(csv_file_name):
    
    dataset = pd.read_csv(csv_file_name, header = 0, index_col = 0)
    values  = dataset.values
    
    scaler  = MinMaxScaler(feature_range=(0, 1))
    data    = scaler.fit_transform(np.double(values[0:1040, 0:9]))
    '''
    Train_X = data[0:218, 0:8]
    Train_y = data[0:218, 8:9]

    Test_X  = data[219:231, 0:8]
    Test_y  = data[219:231, 8:9]
    '''
    '''
    Train_X = data[0:206, 0:8]
    Train_y = data[0:206, 8:9]

    Test_X  = data[207:231, 0:8]
    Test_y  = data[207:231, 8:9]
    '''
    #'''
    Train_X = data[0:800, 0:8]
    Train_y = data[0:800, 8:9]

    Test_X  = data[801:1000, 0:8]
    Test_y  = data[801:1000, 8:9]
    #'''

    return Train_X, Train_y, Test_X, Test_y








