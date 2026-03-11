import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Set experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Experiment")

# 2️⃣ Start MLflow run
with mlflow.start_run():

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model parameters
    C = 0.5
    max_iter = 200

    # Train model
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # 3️⃣ Log parameters
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)

    # 4️⃣ Log metric
    mlflow.log_metric("accuracy", accuracy)

    # 5️⃣ Log model
    mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="Iris_Model"
)

    print("Accuracy:", accuracy)