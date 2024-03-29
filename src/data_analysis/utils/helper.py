import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_analysis.utils.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    """
    Evaluates a set of regression models using GridSearchCV and calculates various metrics.

    Args:
        X_train (pd.DataFrame or np.array): Training features.
        X_test (pd.DataFrame or np.array): Testing features.
        y_train (pd.Series or np.array): Training target values.
        y_test (pd.Series or np.array): Testing target values.
        models (dict): Dictionary of model names and their corresponding estimators.
        params (dict): Dictionary of parameter grids for each model.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each model.
    """

    results = []
    for model_name, model in models.items():
        search = GridSearchCV(model, params[model_name], cv=2, scoring='neg_mean_squared_error', verbose=0)
        search.fit(X_train, y_train)  # Main change: Fit directly 

        # Calculate metrics (using the best estimator)
        y_train_pred = search.predict(X_train)
        y_test_pred = search.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Store results
        results.append({
            'Model': model_name,
            'Best Parameters': search.best_params_,
            'Train MSE': train_mse,
            'Train RMSE': train_rmse,
            'Train MAE': train_mae,
            'Train R2': train_r2,
            'Test MSE': test_mse,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R2': test_r2
        })
    return pd.DataFrame(results), train_mse, test_mse, test_mae, train_mae, test_r2, train_r2

def get_best_model(results_df, models, X_train, y_train):
    """
    Finds the best model based on evaluation metrics.

    Args:
        results_df (pd.DataFrame): DataFrame containing model evaluation results.
                                Assumes it has a 'Test MSE' column.

    Returns:
        tuple: Containing the following:
            - best_model_name (str): Name of the best performing model.
            - best_model_score (float): Test MSE of the best model.
            - best_model_params (dict): Parameter set of the best model.
            - best_model (estimator): The fitted best model object.
    """

    # Find best model based on Test MSE
    best_model_index = results_df['Test MSE'].idxmin()
    best_model_name = results_df.loc[best_model_index, 'Model']
    best_model_score = results_df.loc[best_model_index, 'Test MSE']

    # Extract best parameters and refit model
    best_model_params = results_df.loc[best_model_index, 'Best Parameters']  # Assuming you added this column
    best_model = models[best_model_name].set_params(**best_model_params)
    best_model.fit(X_train, y_train)  # Refit the best model on entire training set   # Refit the best model on entire training set

    return best_model_name, best_model_score, best_model_params, best_model

def save_best_model(filename, best_model):
    """Saves the best model and its parameters to a pickle file.

    Args:
        filename (str): Name of the pickle file to save (include '.pkl' extension).
        best_model: The fitted best model object.
        best_params (dict): Dictionary containing the best parameters.
    """

    with open(filename, 'wb') as f:
        pickle.dump(best_model, f)


