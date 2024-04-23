import os, sys
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.utils.utils import save_object


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
  def __init__(self):
    self.data_ingestion = DataIngestion()
    self.data_transformation = DataTransformation()
    self.model_trainer = ModelTrainer()
    self.model_evaluation = ModelEvaluation()

  def start_data_ingestion(self):
    try:
      
      train_data, test_data = self.data_ingestion.initiate_data_ingestion()
      
      train_arr, test_arr = self.data_transformation.initiate_data_transformation(train_data, test_data)
      
      print(self.model_trainer.initate_model_training(train_arr, test_arr))

      self.model_evaluation.initiate_model_evaluation(test_arr)

    except Exception as e:
      logging.info("Exception occured at TrainingPipeline")
      raise DetailedError(e)
    
  
if __name__ == "__main__":
  obj = TrainingPipeline()
  obj.start_data_ingestion()