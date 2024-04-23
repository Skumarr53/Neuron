import os, sys
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.utils.utils import save_object, load_object

import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluation:
  def __init__(self):
    pass

  def eval_metric(self, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2
  
  def initiate_model_evaluation(self,test_array):
    try:
      X_test, y_test = test_array[:, :-1], test_array[:, -1]
      model = load_object("artifacts/model.pkl")

      mlflow.set_registry_uri("")
      tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

      with mlflow.start_run():
        pred = model.predict(X_test)
        rmse, mae, r2 = self.eval_metric(y_test, pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        if tracking_url_type_store != "file":
          mlflow.sklearn.log_model(model, "model", registered_model_name="model")
        else:
          mlflow.sklearn.log_model(model, "model")

    except Exception as e:
      raise DetailedError(e, sys)