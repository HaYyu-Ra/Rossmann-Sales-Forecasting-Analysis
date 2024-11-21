import os
import pickle
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import logging
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress TensorFlow warnings related to oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize FastAPI app
app = FastAPI()

# Configure static files and templates for serving HTML
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the models
model_dir = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/Rossmann_Sales_Forecasting_Project/models/saved_models'

# Load LSTM model with the updated path
lstm_model_path = os.path.join(model_dir, 'lstm_model_21-11-2024-09-57-37.h5')
lstm_model = load_model(lstm_model_path)

# Fixing the compiled model warning (only if necessary for predictions)
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Load MinMaxScaler with the updated path
scaler_path = os.path.join(model_dir, 'minmax_scaler_21-11-2024-09-57-37.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load Random Forest model with the updated path
rf_model_path = os.path.join(model_dir, 'random_forest_model_21-11-2024-09-57-37.pkl')

# Check for scikit-learn version mismatch and advise user
current_version = pd.__version__
required_version = '1.5.1'  # Assuming model was saved with version 1.5.1
if current_version != required_version:
    warnings.warn(f"InconsistentVersionWarning: Trying to unpickle estimator with scikit-learn version {required_version}, but your current version is {current_version}.")
    warnings.warn("It's advised to either downgrade scikit-learn or retrain the model.")

with open(rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)

# Root route to serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint for LSTM predictions
@app.post("/predict/lstm")
async def predict_lstm(input_data: dict):
    try:
        # Validate input
        input_list = input_data.get("input")
        if not isinstance(input_list, list):
            raise ValueError("Input must be a list of numbers.")

        # Prepare data
        input_array = np.array(input_list).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)

        # Prepare sequences for LSTM
        if len(scaled_input) < 30:
            raise ValueError("At least 30 data points are required for LSTM prediction.")
        X_lstm = [scaled_input[i-30:i, 0] for i in range(30, len(scaled_input))]
        X_lstm = np.array(X_lstm).reshape((len(X_lstm), 30, 1))

        # Make prediction
        lstm_prediction = lstm_model.predict(X_lstm[-1].reshape(1, 30, 1))
        response_data = {"prediction": lstm_prediction.tolist()}
        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

# API endpoint for Random Forest predictions
@app.post("/predict/rf")
async def predict_rf(input_data: dict):
    try:
        # Validate input
        input_list = input_data.get("input")
        if not isinstance(input_list, list) or not all(isinstance(row, list) for row in input_list):
            raise ValueError("Input must be a list of lists.")

        # Prepare data
        input_df = pd.DataFrame(input_list)
        prediction = rf_model.predict(input_df)
        response_data = {"prediction": prediction.tolist()}
        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
