# PeakOne K-Means Model - Deployment Guide

## ✅ Training Complete

Your model has been successfully trained with the following results:

- **Total Customers**: 2,000
- **Optimal Clusters**: 3
- **Best Silhouette Score**: 0.0905

### Customer Segments Identified:
1. **Digital Professionals** - 1,142 customers (57%)
2. **Loan Dependent Customers** - 743 customers (37%)
3. **High Net Worth Customers** - 115 customers (6%)

### Model Files Created:
- ✅ `peakone_kmeans_model.pkl` - Trained K-Means model
- ✅ `peakone_scaler.pkl` - Feature scaler
- ✅ `peakone_label_encoders.pkl` - Categorical encoders
- ✅ `peakone_customers_segmented.csv` - Segmented customer data

---

## 🚀 Deploy to Render

### Step 1: Test Locally (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python main.py
```

Visit `http://localhost:8000/docs` to test the API

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: PeakOne K-Means API"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Render will auto-detect the `render.yaml` configuration
5. Click **"Create Web Service"**

**Important**: Make sure these files are in your repo:
- `peakone_kmeans_model.pkl`
- `peakone_scaler.pkl`
- `peakone_label_encoders.pkl`

### Step 4: Configure Environment (if needed)

Render will automatically use:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Python Version**: 3.11.0

### Step 5: Test Your Deployed API

Once deployed, visit:
- `https://your-app.onrender.com/` - Health check
- `https://your-app.onrender.com/docs` - Interactive API docs
- `https://your-app.onrender.com/segments` - View all segments

---

## 📡 API Endpoints

### GET /
Health check and API status

### GET /health
Detailed health status with model loading info

### POST /predict
Predict customer segment

**Example Request:**
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

**Example Response:**
```json
{
  "cluster": 1,
  "segment_name": "Digital Professionals",
  "confidence": 0.856
}
```

### GET /segments
List all available customer segments

---

## 🧪 Test the API

### Using cURL:
```bash
curl -X POST "https://your-app.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python:
```python
import requests

url = "https://your-app.onrender.com/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

---

## 📊 Model Performance

- **Silhouette Score**: 0.0905 (k=3)
- **Features Used**: 16 engineered features
- **Algorithm**: K-Means with 20 initializations
- **Convergence**: 300 max iterations

---

## 🔧 Troubleshooting

### Model files not loading on Render:
- Ensure `.pkl` files are committed to Git (not in `.gitignore`)
- Check file sizes (GitHub has 100MB limit per file)
- Consider using Git LFS for large files

### API returns 503 error:
- Check Render logs for model loading errors
- Verify all dependencies are in `requirements.txt`
- Ensure Python version compatibility

### Predictions seem incorrect:
- Verify input data matches training data format
- Check that categorical values are recognized
- Review the segmented CSV for expected patterns

---

## 📝 Next Steps

1. ✅ Model trained and ready
2. ⬜ Push to GitHub
3. ⬜ Deploy to Render
4. ⬜ Test API endpoints
5. ⬜ Integrate with your application

**Your API is ready for deployment!** 🎉
