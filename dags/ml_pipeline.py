from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    "ml_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    clean = BashOperator(
        task_id="clean",
        bash_command="python /opt/airflow/dags/src/data-preprocessing.py --input /opt/airflow/dags/data/raw/data.csv --output /opt/airflow/dags/data/processed/cleaned_data.csv",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="python /opt/airflow/dags/src/feature-engineer.py --input /opt/airflow/dags/data/processed/cleaned_data.csv --output /opt/airflow/dags/data/processed/features.csv --preprocessor /opt/airflow/dags/src/preprocessor.pkl",
    )

    train = BashOperator(
        task_id="train",
        bash_command="python /opt/airflow/dags/src/train.py --config /opt/airflow/dags/src/model_config.yaml --data /opt/airflow/dags/data/processed/features.csv --models-dir /opt/airflow/dags/models --dvc",
    )

    clean >> feature_engineering >> train
