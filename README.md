******MLOps Pipeline: DVC + MLflow + Airflow + GitHub Actions + Docker + Minikube******

****🚀 Overview****
This repository implements a robust MLOps pipeline combining:
DVC: Data versioning 
MLflow: Experiment tracking
Apache Airflow: Automated ML training and retraining upon data drift
Docker: Containerize application
GitHub Actions: CI/CD for Dockerization and deployment
Minikube: Local Kubernetes cluster for model deployment

****🛠️ Prerequisites****
Python
DVC
Apache-airflow
MLflow
Docker
Minikube
kubectl
GitHub account

**🛠️ Setting Up DVC with Local Storage for Data Versioning**
dvc init
dvc remote add -d myremote /path/to/storage
dvc add data/dataset.csv
git add data/dataset.csv.dvc .gitignore
git commit -m "Add dataset.csv to DVC tracking"
dvc push
dvc pull

**🐧Running Airflow on Windows with WSL for MLflow tracked Training**
sudo apt update
sudo apt install python3-pip
pip3 install apache-airflow
airflow db init
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
airflow webserver -p 8080
airflow scheduler
Open your web browser and go to http://localhost:8080 to access the Airflow web interface.

**🚀 GitHub Actions Workflow Deployment**
Push changes to the main branch
