from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import tensorflow as tf
import joblib
from PIL import Image
import numpy as np

# Reemplace esto con su implementación:
class ApiInput(BaseModel):
    features: str

# Reemplace esto con su implementación:
class ApiOutput(BaseModel):
    forecast: float

app = FastAPI()
model = joblib.load("model.joblib")

# Reemplace esto con su implementación:
@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    model = joblib.load("model.joblib")
    image_a= np.array(Image.open(data.features).resize((300,300))).reshape((1, 300, 300, 3))
    pred = round(float(model.predict(image_a)))
    prediction = ApiOutput(forecast=pred)
    return prediction
