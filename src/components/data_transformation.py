import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError
from dataclasses import dataclass

import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.utils import save_object
from pdb import set_trace

@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()

  def get_data_transformer_object(self):
    try:
      categorical_columns = ['cut', 'color', 'clarity']
      numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

      num_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
      )

      cat_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinalencoder', OrdinalEncoder()),
            ('scaler', StandardScaler(with_mean=False))
        ])
      
      preprocessor = ColumnTransformer(
        [
            ('num_pipeline', num_pipeline, numerical_columns),
            ('cat_pipeline', cat_pipeline, categorical_columns)
        ]
      )

      return preprocessor
    except Exception as e:
      logging.info("Exception occured in the initiate_datatransformation")
      raise DetailedError(e)
    

  def initiate_data_transformation(self, train_path, test_path):
      try:
          train_df = pd.read_csv(train_path)
          test_df = pd.read_csv(test_path)

          logging.info("Read train and test data completed")
          os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

          logging.info("Obtaining preprocessing object")

          preprocessing_obj = self.get_data_transformer_object()

          target_column_name = 'price'
          drop_columns = [target_column_name, 'id']
          # set_trace()

          target_feature_train_df = train_df[target_column_name]
          input_feature_train_df = train_df.drop(labels=drop_columns, axis=1)

          target_feature_test_df = test_df[target_column_name]
          input_feature_test_df = test_df.drop(labels=drop_columns, axis=1)

          logging.info("Applying preprocessing object on training dataframe and testing dataframe")

          input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
          input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

          train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
          test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

          save_object(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path)

          logging.info("Saved preprocessing object")

          return (
              train_arr,
              test_arr,
          )
      except Exception as e:
          logging.info("Exception occured in the initiate_datatransformation")
          raise DetailedError(e)
      

if __name__ == "__main__":
    obj = DataTransformation()
    train_arr, test_arr = obj.initiate_data_transformation(train_path="artifacts/train.csv", test_path="artifacts/test.csv")