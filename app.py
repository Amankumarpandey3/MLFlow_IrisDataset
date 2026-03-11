from pydantic import BaseModel
from fastapi import FastAPI
import mlflow.pyfunc
import mlflow


app = FastAPI()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("model")
class IrisInput(BaseModel):
    data: list

@app.post("/predict")
def predict(input: IrisInput):
    prediction = model.predict(input.data)
    return {"prediction": prediction.tolist()}