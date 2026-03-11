from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Optional
import os

app = FastAPI(
    title="PeakOne Bank Customer Segmentation API",
    description="K-Means clustering API for customer segmentation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
try:
    kmeans_model = joblib.load("peakone_kmeans_model.pkl")
    scaler = joblib.load("peakone_scaler.pkl")
    label_encoders = joblib.load("peakone_label_encoders.pkl")
    print("Model artifacts loaded successfully")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    kmeans_model = None
    scaler = None
    label_encoders = None

# Segment names mapping
SEGMENT_NAMES = {
    0: "High Net Worth Customers",
    1: "Digital Professionals",
    2: "Loan Dependent Customers",
    3: "Mass Market Customers",
    4: "Low Engagement Customers",
    5: "Emerging Professionals",
    6: "Young Digital Customers",
    7: "Savings Focused Customers",
    8: "Credit Builders",
    9: "Inactive Customers"
}

class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=70, description="Customer age")
    salary_ngn: float = Field(..., gt=0, description="Annual salary in NGN")
    account_balance_ngn: float = Field(..., ge=0, description="Account balance in NGN")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    tenure_years: float = Field(..., ge=0, description="Years with bank")
    monthly_transactions: int = Field(..., ge=0, description="Number of monthly transactions")
    avg_transaction_value_ngn: float = Field(..., ge=0, description="Average transaction value in NGN")
    digital_engagement_score: float = Field(..., ge=0, le=100, description="Digital engagement score")
    gender: str = Field(..., description="Gender (Male/Female)")
    state: str = Field(..., description="State of residence")
    occupation: str = Field(..., description="Occupation")
    borrowing_history: str = Field(..., description="Borrowing history")
    products: str = Field(..., description="Comma-separated list of products")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "salary_ngn": 5000000,
                "account_balance_ngn": 2500000,
                "credit_score": 720,
                "tenure_years": 5.5,
                "monthly_transactions": 25,
                "avg_transaction_value_ngn": 50000,
                "digital_engagement_score": 75,
                "gender": "Male",
                "state": "Lagos",
                "occupation": "Engineer",
                "borrowing_history": "Good",
                "products": "Savings,Current,Credit Card"
            }
        }

class PredictionResponse(BaseModel):
    cluster: int
    segment_name: str
    confidence: Optional[float] = None

@app.get("/")
def read_root():
    return {
        "message": "PeakOne Bank Customer Segmentation API",
        "status": "active",
        "model_loaded": kmeans_model is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": kmeans_model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_segment(customer: CustomerData):
    if kmeans_model is None or scaler is None or label_encoders is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create DataFrame from input
        df = pd.DataFrame([customer.dict()])
        
        # Feature engineering
        df["income_balance_ratio"] = df["account_balance_ngn"] / (df["salary_ngn"] + 1)
        df["transaction_volume"] = df["monthly_transactions"] * df["avg_transaction_value_ngn"]
        
        # Age group segmentation
        age_bins = [18, 25, 35, 45, 55, 70]
        age_labels = ["GenZ", "YoungAdult", "MidCareer", "SeniorProfessional", "PreRetirement"]
        df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
        
        # Product count
        df["product_count"] = df["products"].apply(lambda x: len(str(x).split(",")))
        
        # Encode categorical variables
        categorical_columns = ["gender", "state", "occupation", "borrowing_history", "age_group"]
        for col in categorical_columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen labels
                    df[col] = 0
        
        # Select features in correct order
        features = [
            "age", "salary_ngn", "account_balance_ngn", "credit_score",
            "tenure_years", "monthly_transactions", "avg_transaction_value_ngn",
            "digital_engagement_score", "transaction_volume", "income_balance_ratio",
            "product_count", "gender", "state", "occupation", "borrowing_history", "age_group"
        ]
        
        X = df[features]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict cluster
        cluster = int(kmeans_model.predict(X_scaled)[0])
        
        # Get distances to all cluster centers for confidence
        distances = kmeans_model.transform(X_scaled)[0]
        min_distance = distances[cluster]
        avg_distance = np.mean(distances)
        confidence = float(1 - (min_distance / avg_distance)) if avg_distance > 0 else 0.5
        
        segment_name = SEGMENT_NAMES.get(cluster, f"Cluster {cluster}")
        
        return PredictionResponse(
            cluster=cluster,
            segment_name=segment_name,
            confidence=round(confidence, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/segments")
def get_segments():
    return {
        "segments": SEGMENT_NAMES,
        "total_clusters": len(SEGMENT_NAMES)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
