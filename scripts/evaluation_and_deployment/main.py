from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np

# Reemplace esto con su implementación:
class ApiInput(BaseModel):
    features: np.ndarray
    class Config:
        arbitrary_types_allowed = True

# Reemplace esto con su implementación:
class ApiOutput(BaseModel):
    forecast: float

app = FastAPI()
#model = joblib.load("model.joblib")
model = load_model('weights.h5')

# Reemplace esto con su implementación:
@app.post("/predict")
async def predict(data: ApiInput) -> ApiOutput:
    image_a= data.features.reshape((1, 300, 300, 3))
    pred = round(float(model.predict(image_a)))
    prediction = ApiOutput(forecast=pred)
    return prediction
