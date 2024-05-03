import os
import pandas as pd
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.utils.utils import load_object


class BatchPredictionPipeline:
    def __init__(self):
        # Load preprocessor and model artifacts
        self.preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
        self.model = load_object(os.path.join("artifacts", "model.pkl"))

    def load_data(self, filepath):
        # Load data from a CSV file
        try:
            logging.info("Loading data from file: {}".format(filepath))
            return pd.read_csv(filepath)
        except Exception as e:
            raise DetailedError(e)

    def predict(self, data):
        # Perform predictions on the batch data
        try:
            # Preprocess the data
            logging.info("Preprocessing the data")
            scaled_data = self.preprocessor.transform(data)

            # Predict using the model
            logging.info("Making predictions")
            predictions = self.model.predict(scaled_data)

            return predictions
        except Exception as e:
            raise DetailedError(e)

    def save_predictions(self, data, predictions, output_filepath):
        # Save the predictions into a new CSV file
        try:
            logging.info("Saving predictions to file: {}".format(output_filepath))
            data['predictions'] = predictions
            data.to_csv(output_filepath, index=False)
        except Exception as e:
            raise DetailedError(e)

    def run_batch_prediction(self, input_filepath, output_filepath):
        # Main method to execute the batch prediction process
        try:
            data = self.load_data(input_filepath)
            predictions = self.predict(data)
            self.save_predictions(data, predictions, output_filepath)
        except Exception as e:
            logging.error("Error in running batch prediction: {}".format(e))
            raise DetailedError(e)


# Example of how to use the BatchPredictionPipeline
if __name__ == "__main__":
    pipeline = BatchPredictionPipeline()
    pipeline.run_batch_prediction("artifacts/raw.csv", "artifacts/predicted.csv")
