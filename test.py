            # mlflow.set_registry_uri("https://dagshub.com/ManojShendre1202/mlflow.mlflow")
            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # # mlflow

            # with mlflow.start_run():

            #     predicted_qualities = best_model.predict(X_test)

            #     (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

            #     mlflow.log_params(best_params)

            #     mlflow.log_metric("rmse", rmse)
            #     mlflow.log_metric("r2", r2)
            #     mlflow.log_metric("mae", mae)


            #     # Model registry does not work with file store
            #     if tracking_url_type_store != "file":

            #         # Register the model
            #         # There are other ways to use the Model Registry, which depends on the use case,
            #         # please refer to the doc for more information:
            #         # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #         mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
            #     else:
            #         mlflow.sklearn.log_model(best_model, "model")

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import comet_ml  

# Replace with your Comet.ml credentials
comet_api_key = "cM9zE9jmvHCBKH36N8AICfO2x"  
project_name = "end-to-end-data-analysis"
experiment_name = "regression-experiment"

experiment = comet_ml.Experiment(
    api_key=comet_api_key, project_name=project_name, workspace="manojshendre1202"
)
experiment.set_name(experiment_name)  

# Load and Prepare Dataset
boston_data = fetch_california_housing()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
df['target'] = boston_data.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics to Comet ML
experiment.log_metric("MSE", mse)
experiment.log_metric("R2", r2)

# Create a scatter plot
plt.figure(figsize=(8,6)) 
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predictions vs. Actual")

# Log the plot to Comet ML
experiment.log_figure(figure_name = "Predictions vs Actual", figure=plt)

# Log hyperparameters (if you have any)
experiment.log_parameters({"random_state": 42})  

# Save the model with Comet ML if desired:
# experiment.log_model("regression-model", "model.pkl")

experiment.end()
