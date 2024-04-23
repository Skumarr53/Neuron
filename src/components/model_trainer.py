import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object, evaluate_model, find_best_model_by_metric

from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
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
                'Elasticnet': ElasticNet()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_name, best_score = find_best_model_by_metric(model_report, 'r2')

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {round(best_score,2)}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {round(best_score,2)}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models[best_model_name]
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise DetailedError(e)
        

if __name__ == '__main__':
    obj = ModelTrainer()
    obj.initate_model_training(train_array=X_train, test_array=X_test)