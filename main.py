from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

app = FastAPI()

# CORS (for Bubble)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

model = pickle.load(open(os.path.join(os.getcwd(), "model.pkl"), "rb"))

# Home
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# POST (Bubble)
@app.post("/predict")
def predict_post(data: dict):
    try:
        bmi = data.get("bmi")

        if bmi is None:
            return {"error": "Please provide 'bmi'"}

        features = np.array([[bmi]])  # only 1 input

        prediction = model.predict(features)

        return {
            "prediction": float(prediction[0]),
            "bmi": bmi
        }

    except Exception as e:
        return {"error": str(e)}

# GET (browser test)
@app.get("/predict")
def predict_get(bmi: float):
    try:
        features = np.array([[bmi]])

        prediction = model.predict(features)

        return {
            "prediction": float(prediction[0]),
            "bmi": bmi
        }

    except Exception as e:
        return {"error": str(e)}