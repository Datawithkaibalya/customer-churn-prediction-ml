import sys
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Add src folder to system path to load preprocessing tools
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_processing import preprocess_data

app = FastAPI(title="Production Customer Churn Retention API")

class CustomerDataInput(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str  # Must pass "Month-to-month", "One year", or "Two year"
    internet_service: str  # Must pass "DSL", "Fiber optic", or "No"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

@app.post("/predict")
def predict_churn(inp: CustomerDataInput):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Machine learning model configuration error. Run train.py first."}
    
    # Load model binaries
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # Convert incoming dictionary data payload to Dataframe
    raw_data = pd.DataFrame({k: [v] for k, v in inp.model_dump().items()})
    processed_features = preprocess_data(raw_data, is_training=False)
    
    # Compute outputs
    prediction = int(model.predict(processed_features)[0])
    probability = float(model.predict_proba(processed_features)[0][1])
    
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "status": "High Risk of Churn" if prediction == 1 else "Loyal Customer"
    }
