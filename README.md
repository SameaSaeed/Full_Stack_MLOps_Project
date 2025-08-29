minikube start --driver=docker
minikube -p minikube docker-env | Invoke-Expression
docker build -t model-api:latest .
kubectl apply -f K8s/deployment.yaml
kubectl apply -f K8s/service.yaml
kubectl rollout status deployment/model-api --timeout=120s
kubectl port-forward svc/model-api-service 8000:80
