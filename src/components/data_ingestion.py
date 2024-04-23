import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError


import os, sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
  train_data_path: str = os.path.join("artifacts", "train.csv")
  test_data_path: str = os.path.join("artifacts", "test.csv")
  raw_data_path: str = os.path.join("artifacts", "data.csv")




class DataIngestion:
  def __init__(self):
    self.ingestion_config = DataIngestionConfig()

  def initiate_data_ingestion(self):
    logging.info("data ingestion started")
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/abhijitpaul0212/GemstonePricePrediction/master/artifacts/raw.csv")
        logging.info("Read the dataset as dataframe")

        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

        logging.info("Train test split initiated")
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        train_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
    except Exception as e:
        logging.info("Exception occured at Data Ingestion stage")
        raise DetailedError(e)
    

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()