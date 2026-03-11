import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.artifacts.download_artifacts(
    artifact_uri="models:/Iris_Model@production",
    dst_path="model"
)