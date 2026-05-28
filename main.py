from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

class ApiInput(BaseModel):
    features: List[float]

class ApiOutput(BaseModel):
    forecast: float

app = FastAPI()
model = joblib.load("model.joblib")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    features = np.array(data.features).reshape(1, -1)
    result = model.predict(features)[0]
    return ApiOutput(forecast=float(result))
