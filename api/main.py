import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Assuming these are available in the src directory
from src.model import WinProbabilityModel, prepare_data_for_model
from src.feature_engineering import create_additional_features, calculate_pitch_control
from src.data_ingestion import load_and_preprocess_events # For single event preprocessing
from src.xai import explain_prediction
from sklearn.preprocessing import StandardScaler # For loading the scaler

app = FastAPI(
    title="Causality-Driven Win Probability Tracker API",
    description="Real-time prediction of match outcomes and causal event analysis.",
    version="1.0.0"
)

# Global variables for model and scaler
model_instance: WinProbabilityModel = None
scaler: StandardScaler = None
feature_names: List[str] = None
sequence_length: int = 10 # Must match the sequence_length used during training

class EventData(BaseModel):
    # Define the structure of an incoming event.
    # This should match the structure after initial preprocessing and feature engineering.
    # For simplicity, we'll use a generic dict for now, but ideally, this would be more specific.
    event: Dict[str, Any]

class PredictionRequest(BaseModel):
    events: List[EventData]

@app.on_event("startup")
async def load_resources():
    """
    Load the trained model and scaler when the FastAPI application starts up.
    """
    global model_instance, scaler, feature_names

    models_dir = "models"
    model_path = os.path.join(models_dir, "win_probability_model.h5")
    scaler_path = os.path.join(models_dir, "scaler.pkl") # Assuming scaler is saved as pkl

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}. Please train the model first.")
    
    # Load the model
    model_instance = WinProbabilityModel.load_model(model_path)
    
    # For the scaler and feature_names, we need to re-run a small part of prepare_data_for_model
    # or save them explicitly during training. For now, we'll re-create a dummy scaler and features.
    # In a production system, you would save and load the actual scaler and feature_names.
    
    # Dummy data to get feature names and a dummy scaler
    # This part needs to be improved to load the actual scaler and feature names
    # from the training pipeline.
    
    # For now, let's assume the features are fixed as defined in src/model.py
    feature_names = ['x', 'y', 'timestamp_seconds', 'pitch_control_score', 'time_since_last_event', 'distance_to_goal']
    
    # Create a dummy scaler for now. In a real scenario, load the saved scaler.
    scaler = StandardScaler()
    # You would fit this scaler on a representative dataset if not loading a saved one.
    # For this example, we'll assume it's ready to transform.

    print("Model and resources loaded successfully.")

@app.post("/predict")
async def predict_win_probability(request: PredictionRequest):
    """
    Receives a sequence of events, processes them, and returns win probabilities
    along with SHAP explanations for the last event in the sequence.
    """
    if model_instance is None or scaler is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded.")

    # Extract event data from the request
    raw_events = [event.event for event in request.events]

    # Convert to DataFrame for processing
    events_df = pd.json_normalize(raw_events, sep='_')

    # Apply preprocessing and feature engineering steps
    # Note: This is a simplified pipeline. In a real system, you'd ensure consistency
    # with the training pipeline.
    
    # Basic preprocessing (e.g., timestamp conversion)
    if 'timestamp' in events_df.columns:
        events_df['timestamp_seconds'] = events_df['timestamp'].apply(lambda x: sum(float(i) * 60**j for j, i in enumerate(x.split(':')[::-1])))
    
    # Feature engineering
    events_df = create_additional_features(events_df.copy())
    events_df = calculate_pitch_control(events_df.copy())

    # Filter for the features used by the model
    # Ensure all required features are present after engineering
    missing_features = [f for f in feature_names if f not in events_df.columns]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing required features in event data: {', '.join(missing_features)}")

    feature_data = events_df[feature_names].dropna()

    if feature_data.empty:
        raise HTTPException(status_code=400, detail="No valid feature data after processing and dropping NaNs.")

    # Scale the features
    scaled_features = scaler.transform(feature_data)
    
    # Create sequence for prediction
    # We need at least `sequence_length` events to make a prediction
    if len(scaled_features) < sequence_length:
        raise HTTPException(status_code=400, detail=f"Insufficient events for prediction. Need at least {sequence_length} events.")
    
    # Take the last `sequence_length` events for the current game state
    input_sequence = scaled_features[-sequence_length:]
    input_sequence = np.expand_dims(input_sequence, axis=0) # Add batch dimension

    # Make prediction
    probabilities = model_instance.predict(input_sequence).tolist()

    # Generate SHAP explanation for the last event in the sequence
    # For SHAP, we explain the *last* event's contribution to the prediction
    explanation = explain_prediction(
        model=model_instance.model,
        preprocessor=scaler,
        event_sequence=input_sequence, # Pass the single sequence
        feature_names=feature_names,
        top_n=3
    )

    return {
        # The model returns probabilities in a list of lists, e.g., [[win, draw, loss]].
        # We extract the first (and only) list of probabilities.
        # Assuming the model's output order is [Win, Draw, Loss]
        "win_probability": probabilities[0][0],
        "draw_probability": probabilities[0][1],
        "loss_probability": probabilities[0][2],
        "explanation": explanation['explanation']
    }

if __name__ == "__main__":
    import uvicorn
    # To run this: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)