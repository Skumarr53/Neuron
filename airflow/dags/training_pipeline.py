from __future__ import annotations
import os, json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.utils import upload_folder_to_blob



with DAG(
  'Gem_training_pipeline',
  default_args={
    'retries': 0,
    'retry_delay': pendulum.Duration(minutes=1),
  },
  description="it is my training pipeline",
  schedule="@hourly",
  start_date=pendulum.datetime(2024, 4, 24, tz="UTC"),
  catchup=False,
    tags=["machine_learning ", "classification","gemstone"],
) as dag:
  
  dag.doc_md = __doc__

  def training_ingestion(**kwargs):
    ti = kwargs['ti']
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    ti.xcom_push("data_ingestion_artifact", {"_train_path": train_path, "_test_path": test_path})
  
  def training_tranformation(**kwargs):
    ti = kwargs['ti']
    data_transformation = DataTransformation()
    data_ingest_artifact = ti.xcom_pull(key="data_ingestion_artifact", task_ids="training_ingestion")
    train_path, test_path = data_ingest_artifact["_train_path"], data_ingest_artifact["_test_path"]
    data_transformation.initiate_data_transformation(train_path, test_path)

  def training_model(**kwargs):
    ti = kwargs['ti']
    model_trainer = ModelTrainer()
    art_path = model_trainer.initate_model_training()
    ti.xcom_push("trained_artifacts", {"folder_path": art_path})


  def upload_to_blob_storage(**kwargs):
    ti = kwargs['ti']
    trained_art = ti.xcom_pull(key="trained_artifacts", task_ids="training_model")
    storage_account_name = os.environ.get("AZ_STORAGE_ACC_NAME")
    container_name = os.environ.get("AZ_UPLOAD_CONTAINER_NAME")
    trained_art_path = trained_art['folder_path']
    upload_folder_to_blob(storage_account_name, container_name, trained_art_path)


  data_ingestion_task = PythonOperator(
    task_id="training_ingestion",
    python_callable=training_ingestion
  )

  data_transformation_task = PythonOperator(
    task_id="training_tranformation",
    python_callable=training_tranformation
  )

  model_trainer_task = PythonOperator(
    task_id="training_model",
    python_callable=training_model
  )

  artifact_upload_task = PythonOperator(
      task_id="artifacts_upload",
      python_callable=upload_to_blob_storage
  )

  data_ingestion_task.doc_md = dedent(
    """ 
    #### Data Ingestion

    This task will ingest the data from the URL
    and create the train and test dataset
    """
  )

  data_transformation_task.doc_md = dedent(
    """ 
    #### Data Transformation

    This task will transform the data
    """
  )

  model_trainer_task.doc_md = dedent(
    """ 
    #### Model Trainer

    This task will train the model
    """
  )

  artifact_upload_task.doc_md = dedent(
    """ 
    #### Artifact Upload

    This task will upload the artifacts to Azure storage
    """
  )

data_ingestion_task >> data_transformation_task >> model_trainer_task >> artifact_upload_task

