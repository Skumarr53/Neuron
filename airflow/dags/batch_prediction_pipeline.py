import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.utils.utils import download_file_from_blob


with DAG(
  'Gem_batch_prediction_pipeline',
  default_args={
    'retries': 2,
    'retry_delay': pendulum.Duration(minutes=5),
  },
  description="it is my batch prediction pipeline",
  schedule="@hourly",
  start_date=pendulum.datetime(2024, 4, 26, tz="UTC"),
  catchup=False,
    tags=["machine_learning ", "classification","gemstone"],
) as dag:
  
dag.doc_md = __doc__

def download_from_blob():
  ti = kwargs['ti']
  storage_account_name = os.environ.get("AZ_STORAGE_ACC_NAME")
  container_name = os.environ.get("AZ_UPLOAD_CONTAINER_NAME")
  download_file_from_blob(storage_account_name, container_name, local_file_path)
