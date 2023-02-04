

# 2023-2-3 written by H.Zhang.



import numpy
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error # MAE
from sklearn.metrics import r2_score            # R2


# ------------------------------------------------------------------------------------- MAPE & SMAPE.

def mape(y_true, y_pred):
    
    return numpy.mean(numpy.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    
    return 2.0 * numpy.mean(numpy.abs(y_pred - y_true) / (numpy.abs(y_pred) + numpy.abs(y_true))) * 100


# ------------------------------------------------------------------------------------- All Metrics.

def compute_metrics(Method, y_true, y_pred):

    MSE  = mean_squared_error(y_true,y_pred)
    MAE  = mean_absolute_error(y_true,y_pred)
    RMSE = numpy.sqrt(mean_squared_error(y_true,y_pred))
    R2   = r2_score(y_true,y_pred)

    MAPE = mape(y_true, y_pred)
    SMAPE= smape(y_true, y_pred)

    print("| ========================================== %s ========================================== |"%(Method))
    
    print('%20s | %10f'%("MSE",  MSE))
    print('%20s | %10f'%("MAE",  MAE))
    print('%20s | %10f'%("RMSE", RMSE))

    print('%20s | %10f'%("R2 Score", R2))
    print('%20s | %10f'%("MAPE", MAPE))
    print('%20s | %10f'%("SMAPE", SMAPE))




    return MSE, MAE, RMSE, R2, MAPE, SMAPE




