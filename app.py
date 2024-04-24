import os
import sys
import json
import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import DetailedError
from src.utils.utils import load_object
from src.pipeline.inference_pipeline import PredictionPipeline,CustomData


from flask import Flask, request, render_template, jsonify

app = Flask(__name__)


@app.route('/')
def home_page():
  return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Endpoint for making predictions. Expects a json with a key 'data'
    that contains a list of dictionaries, each representing a data point.
    """
    try:
      if request.method == 'GET':
          return render_template('form.html')
      else:

        data=CustomData(
          carat=float(request.form.get('carat')),
          depth=float(request.form.get('depth')),
          table=float(request.form.get('table')),
          x=float(request.form.get('x')),
          y=float(request.form.get('y')),
          z=float(request.form.get('z')),
          cut=request.form.get('cut'),
          color=request.form.get('color'),
          clarity=request.form.get('clarity')
        )
        fin_data = data.get_data_as_dataframe()

        # Load the pipeline and make predictions
        pipeline = PredictionPipeline()
        y_pred = pipeline.predict(fin_data)

        # Return the predictions
        return render_template("result.html", final_result=round(y_pred[0], 2))

    except Exception as e:
      logging.error("Error in prediction")
      raise DetailedError(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

