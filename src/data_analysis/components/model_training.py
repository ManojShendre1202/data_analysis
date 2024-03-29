import sys
import os
import pickle
import comet_ml
import numpy as np
from dotenv import load_dotenv
from comet_ml import Experiment
from xgboost import XGBRegressor
from dataclasses import dataclass
from urllib.parse import urlparse
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from data_analysis.utils.logger import logging
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from comet_ml.integration.sklearn import log_model
from data_analysis.utils.exception import CustomException
from sklearn.metrics import mean_squared_error,mean_absolute_error
from data_analysis.utils.helper import save_object, evaluate_models, get_best_model, save_best_model
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

load_dotenv()

comet_api_key = os.getenv('API_KEY')
project_name = os.getenv("project_name")
experiment_name = os.getenv("experiment_name")

experiment = comet_ml.Experiment(
    api_key=comet_api_key, project_name=project_name, workspace=os.getenv("workspace")
)
experiment.set_name(experiment_name) 


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("models","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                    'max_features':['sqrt','log2',None],
                #     'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    # 'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    # 'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    # 'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            results_df, train_mse, test_mse, test_mae, train_mae, test_r2, train_r2 = evaluate_models(X_train, X_test, y_train, y_test, models, params)
            best_model_name, best_model_score, best_model_params, best_model = get_best_model(results_df, models, X_train, y_train) 

            # print(results_df)
            # print("Best Model:", best_model_name)
            # print("Best Test MSE:", best_model_score)
            # print("Best Parameters:", best_model_params)
            # print("Best Estimator Object:", best_model)
            experiment.log_metric("Best Model", best_model)
            experiment.log_metric("Best model parameters", best_model_params) 
            experiment.log_metric("Train MSE", train_mse)
            experiment.log_metric("Test MSE", test_mse)
            experiment.log_metric("Train R2", train_r2)
            experiment.log_metric("Test R2", test_r2)
            experiment.log_metric("Train MAE", train_mae)
            experiment.log_metric("Test MAE", test_mae)

            # Save the best model
            save_best_model('best_model.pkl', best_model) 

            experiment.log_parameters({"random_state": 42}) 

            experiment.end()

        except Exception as e:
            logging.exception(e) 
            raise CustomException(e, sys)
        

