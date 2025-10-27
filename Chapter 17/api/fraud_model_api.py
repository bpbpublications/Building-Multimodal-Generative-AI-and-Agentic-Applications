# === api/fraud_model_api.py ===
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys

app = FastAPI(title="Fraud Detection API")

# Optional: Enable CORS if using from a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check model files before loading
required_files = [
    "model/xgb_model_final.pkl",
    "model/selected_features.pkl",
    "model/scaler.pkl",
    "model/label_encoders.pkl"
]
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing required model artifact: {f}")

# Load model and preprocessors
model = joblib.load("model/xgb_model_final.pkl")
selected_features = joblib.load("model/selected_features.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# Define input schema
class ClaimFeatures(BaseModel):
    HOUR_TO_RAISE_CLAIM: float
    TOTAL_VERIFICATIONS: int
    IS_MISSING_MOBILE: int

@app.post("/predict_fraud")
def predict_fraud(features: ClaimFeatures):
    try:
        input_dict = features.dict()
        df = pd.DataFrame([input_dict])

        for col in label_encoders:
            if col in df.columns:
                le = label_encoders[col]
                df[col] = le.transform(df[col].astype(str))

        df_selected = df[selected_features]
        df_scaled = scaler.transform(df_selected)
        prob = model.predict_proba(df_scaled)[0, 1]

        return {"fraud_probability": round(float(prob), 4)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))