# PeakOne Bank Customer Segmentation API

FastAPI application for K-Means customer segmentation model deployment.

## Features

- Customer segmentation using K-Means clustering
- RESTful API with FastAPI
- Ready for Render deployment
- Automatic feature engineering
- 10 customer segments

## Customer Segments

1. High Net Worth Customers
2. Digital Professionals
3. Loan Dependent Customers
4. Mass Market Customers
5. Low Engagement Customers
6. Emerging Professionals
7. Young Digital Customers
8. Savings Focused Customers
9. Credit Builders
10. Inactive Customers

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (if not already trained):
```bash
python train_model.py
```

3. Run the API:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Health check and API information

### GET /health
Detailed health status

### POST /predict
Predict customer segment

Example request:
```json
{
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
```

### GET /segments
List all available segments

## Render Deployment

1. Push code to GitHub repository

2. Create new Web Service on Render:
   - Connect your repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. Add model files:
   - Upload `peakone_kmeans_model.pkl`
   - Upload `peakone_scaler.pkl`
   - Upload `peakone_label_encoders.pkl`

4. Deploy!

## Testing

Visit `/docs` for interactive API documentation (Swagger UI)
Visit `/redoc` for alternative documentation

## Model Files Required

- `peakone_kmeans_model.pkl` - Trained K-Means model
- `peakone_scaler.pkl` - Feature scaler
- `peakone_label_encoders.pkl` - Label encoders for categorical variables

Run `train_model.py` to generate these files.
