# Python Application for ML-Ops

## Start Prometheus, Grafana, Python Application and ML-Flow UI

```
docker-compose up -d prometheus
docker-compose up -d grafana
docker-compose up -d grafana-dashboards
docker-compose up -d --build python-application
docker exec -it $(docker ps -qf "name=python-application") /bin/sh -c "mlflow ui --host 0.0.0.0 --port 5001"
```

## Visit Browser

```
http://localhost:80   # Python Application
http://localhost:81   # ML-Flow UI
http://localhost:82   # Prometheus
http://localhost:83   # Grafana
```