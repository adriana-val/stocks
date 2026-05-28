from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

class ApiInput(BaseModel):
    features: List[float]

class ApiOutput(BaseModel):
    forecast: float

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    features = [data.features]
    result = model.predict(features)[0]
    return ApiOutput(forecast=float(result))
