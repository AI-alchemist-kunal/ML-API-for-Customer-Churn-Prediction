# project_2_ml_api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os # To check if files exist

# Initialize FastAPI app
app = FastAPI( 
    title="Customer Churn Prediction API",
    description="An API to predict customer churn using a Logistic Regression model."
)

# --- 1. LOAD ARTIFACTS ---
MODEL_PATH = "model.pkl"
COLUMNS_PATH = "training_columns.pkl"
model = None
training_columns = None

if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        training_columns = joblib.load(COLUMNS_PATH)
        print("Model and columns loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
else:
     print(f"Error: '{MODEL_PATH}' or '{COLUMNS_PATH}' not found. "
           "Run model_trainer.py first to generate these files.")

# --- 2. DEFINE INPUT DATA MODEL ---
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 1, "PhoneService": "No", "MultipleLines": "No", "InternetService": "DSL",
                "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
            }
        }

# --- 3. DEFINE API ENDPOINTS ---
@app.get("/")
def read_root():
    if model is None or training_columns is None:
        return {"message": "Welcome! Model artifacts not loaded. Please run model training."}
    return {"message": "Welcome to the Churn Prediction API. Go to /docs for testing."}

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """Predicts customer churn based on input features."""
    if model is None or training_columns is None:
        return {"error": "Model artifacts not loaded. Cannot make predictions."}

    try:
        # Convert Pydantic model to dict, then to DataFrame
        input_data = pd.DataFrame([features.model_dump()])

        # Preprocess: Clean (minimal cleaning needed here as types are enforced by Pydantic)
        input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce').fillna(0) # Ensure numeric
        
        # Preprocess: One-hot encode
        input_df_processed = pd.get_dummies(input_data)
        
        # Preprocess: Reindex to match training columns exactly
        input_df_reindexed = input_df_processed.reindex(columns=training_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df_reindexed)
        probability = model.predict_proba(input_df_reindexed)[0][1] # Churn probability

        prediction_value = int(prediction[0])
        probability_value = float(probability)
        
        return {
            "prediction_label": "Churn" if prediction_value == 1 else "No Churn",
            "prediction_value": prediction_value,
            "churn_probability": probability_value
        }
    except Exception as e:
        # Log the error for debugging in a real application
        print(f"Prediction Error: {e}") 
        return {"error": f"Prediction failed. Input data might be invalid. Details: {str(e)}"}