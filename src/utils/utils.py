import os
from src.exception.exception import DetailedError
from src.logger.logger import logging
import numpy as np
import pickle
from typing import Dict, List, Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


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


def upload_folder_to_blob(storage_account_name, container_name, folder_path):
    try:
        acc_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}"
        # Create a BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url=acc_url, credential=os.environ.get("AZ_BLOB_KEY"))

        # Get a reference to the container
        container_client = blob_service_client.get_container_client('artifacts')

        # Loop through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                # Create a blob client using the file name as the name for the blob
                blob_client = container_client.get_blob_client(blob=file_name)

                # Upload the file to Azure
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                    print(f"File {file_path} uploaded to {file_name}")

    except Exception as e:
        logging.info('Exception occurred')
        raise DetailedError(e)


def download_file_from_blob(storage_account_name, container_name,  local_file_path):
    try:
        account_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}"
        # Create a BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url=account_url, credential=os.environ.get("AZ_BLOB_KEY"))

        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)

        blob_name = os.path.basename(local_file_path)
        # Create a blob client for the specified blob
        blob_client = container_client.get_blob_client(blob=blob_name)

        # Download the blob content
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    except Exception as e:
        logging.info('Exception occurred')
        raise DetailedError(e)