# evaluate_model.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred, X_train):
    """
    Evaluate regression model performance
    """

    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # R-squared
    r2 = r2_score(y_true, y_pred)

    # Adjusted R-squared
    n = X_train.shape[0]   # number of samples
    p = X_train.shape[1]   # number of features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return mse, rmse, mae, r2, adj_r2
