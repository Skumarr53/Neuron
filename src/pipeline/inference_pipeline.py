import os
import sys
import json
import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.utils.utils import load_object
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from src.config import config
from pdb import set_trace


class PredictionPipeline:
  def __init__(self):
    self.preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
    self.metric_name = "r2_score"
    self.model = self.fetch_best_model()

  def fetch_best_model(self,ascending=False):
    """
    Fetch the run ID of the best model from an MLflow experiment based on a specified metric.

    Args:
    experiment_name (str): The name of the MLflow experiment.
    metric_name (str): The name of the metric to sort the models by.
    ascending (bool): Determines if the sorting should be ascending (True) or descending (False).

    Returns:
    str: The run ID of the best model.
    """


    try:
        
        client = MlflowClient()
        experiment_id = client.get_experiment_by_name(config.MLFLOW_EXP).experiment_id
        runs = client.search_runs(
        experiment_ids=[experiment_id],
            order_by=[f"metrics.{self.metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        best_run = runs[0]
        logging.info(f"Best model run ID: {best_run.info.run_id}, {self.metric_name}: {best_run.data.metrics[self.metric_name]}")
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Model loaded successfully")
        return model
    except Exception as e:
        logging.exception(f"Error fetching best model run ID: {e}")
        raise DetailedError(e)
    
  def predict(self, features):
    try:
      scaled_features = self.preprocessor.transform(features)

      prediction = self.model.predict(scaled_features)

      return prediction

    except Exception as e:
      raise DetailedError(e)

class CustomData:
  def __init__(self, carat, depth, table, x, y, z, cut, color, clarity):
    self.data = {'carat': [carat], 'depth': [depth], 'table': [table], 'x': [x], 'y': [y], 'z': [z], 'cut': [cut], 'color': [color], 'clarity': [clarity]}

  def get_data_as_dataframe(self):
    try:
      logging.info("Transforming data into a pandas dataframe")
      return pd.DataFrame(self.data)
    except Exception as e:
      raise DetailedError(e)