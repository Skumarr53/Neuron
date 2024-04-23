import os
from src.exception.exception import DetailedError
from src.logger.logger import logging
import numpy as np
import pickle
from typing import Dict, List, Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(obj, file_path):
    try:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Saving the object using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        
        # logging.INFO(f"File saved successfully at {file_path}")
    except Exception as e:
        logging.exception(f"Error saving file: {e}")
        raise DetailedError(e)

def load_object(file_path):
    try:
        
        # Loading the object using pickle
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.exception(f"Error loading file: {e}")
        raise DetailedError(e)
    


def evaluate(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse, mae

def evaluate_model(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    models: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a list of models on the given data and generate a report.

    Args:
        X_train (np.ndarray): The training feature array.
        y_train (np.ndarray): The training target array.
        X_test (np.ndarray): The test feature array.
        y_test (np.ndarray): The test target array.
        models (Dict[str, Any]): A dictionary of models, where the keys are the names
            of the models and the values are the models themselves.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the R2 scores of the models on
            the test set.
    """
    try:
        report: Dict[str, Dict[str, float]] = {}
        for m_name, mod in models.items():
            # Fit each model to the training data
            model: Any = mod.fit(X_train, y_train)
            
            # Generate predictions on the test data
            y_pred: np.ndarray = model.predict(X_test)
            
            # Evaluate the predictions using various metrics
            r2, mse, mae = evaluate(y_test, y_pred)
        
            report[m_name] = {
                'r2': r2,
                'mse': mse,
                'mae': mae
            }

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise DetailedError(e)
    

def find_best_model_by_metric(models, metric):
    # Initialize variables to store the best model name and the highest score for the given metric
    best_model = None
    best_score = float('-inf')

    # Iterate through each model in the dictionary
    for model_name, metrics in models.items():
        # Check if the current model's score for the given metric is higher than the current best score
        if metrics[metric] > best_score:
            best_model = model_name
            best_score = metrics[metric]

    return best_model, best_score
