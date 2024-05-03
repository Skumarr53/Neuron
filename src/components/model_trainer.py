import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.logger.logger import logging
from src.exception.exception import DetailedError
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import evaluate_model, find_best_model_by_metric

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.config import config

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    preprocessed_train_file_path = os.path.join('artifacts', 'train_transformed.npy')
    preprocessed_test_file_path = os.path.join('artifacts', 'test_transformed.npy')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self):
        try:
            # MLflow Tracking
            mlflow.set_experiment(config.MLFLOW_EXP)
            client = MlflowClient()

            # Load preprocessed data
            train_array = np.load(self.model_trainer_config.preprocessed_train_file_path)
            test_array = np.load(self.model_trainer_config.preprocessed_test_file_path)

            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            best_model_name = None
            best_score = -np.inf

            for model_name, model in models.items():
                with mlflow.start_run():
                    mlflow.log_param("model_name", model_name)
                    # Model training and evaluation
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    mlflow.log_metric("r2_score", score)

                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        mlflow.sklearn.log_model(model, "model")

            logging.info(f'Best Model: {best_model_name} with R2 Score: {best_score}')
        except Exception as e:
            logging.error('Exception occurred during model training', exc_info=True)
            raise DetailedError(e)


if __name__ == '__main__':
    obj = ModelTrainer()
    obj.initiate_model_training()
