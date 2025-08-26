from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='DVC + MLflow pipeline',
    schedule_interval=None,  # manual or trigger-based
    catchup=False,
)

def run_dvc_stage(stage):
    subprocess.run(["dvc", "repro", stage], check=True)

clean_task = PythonOperator(
    task_id='clean',
    python_callable=lambda: run_dvc_stage('clean'),
    dag=dag
)

feature_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=lambda: run_dvc_stage('feature_engineering'),
    dag=dag
)

train_task = PythonOperator(
    task_id='train',
    python_callable=lambda: run_dvc_stage('train'),
    dag=dag
)

# Define order
clean_task >> feature_task >> train_task
