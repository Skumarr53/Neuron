import os
import sys
import json
import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.utils.utils import load_object



class PredictionPipeline:
  def __init__(self):
    self.preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
    self.model = load_object(os.path.join("artifacts", "model.pkl"))

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