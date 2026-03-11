import mlflow
import mlflow.sklearn
import numpy as np
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # SAME AS TRAINING

run_id = "9a3863c1607143a9915e34c58bce8f35"

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Model loaded successfully")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

print("Prediction:", prediction)