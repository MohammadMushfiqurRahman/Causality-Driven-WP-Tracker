import requests
import pandas as pd
import time
import os
import json

# Configuration for the FastAPI endpoint
API_URL = "http://localhost:8000/predict"
PROCESSED_EVENTS_DIR = "data/processed_events"
SEQUENCE_LENGTH = 10 # Must match the model's expected sequence length

def simulate_event_stream(match_event_file, delay_seconds=0.1):
    """
    Simulates a real-time event stream by sending events to the FastAPI endpoint.
    """
    print(f"Starting simulation for match: {match_event_file}")
    
    try:
        events_df = pd.read_csv(match_event_file)
    except FileNotFoundError:
        print(f"Error: Event file not found at {match_event_file}")
        return

    # Sort events by timestamp to ensure chronological order
    if 'timestamp_seconds' in events_df.columns:
        events_df = events_df.sort_values(by='timestamp_seconds').reset_index(drop=True)
    else:
        print("Warning: 'timestamp_seconds' column not found. Events might not be in chronological order.")

    event_buffer = []
    predictions = []

    for index, event in events_df.iterrows():
        event_data = event.to_dict()
        event_buffer.append(event_data)

        # Keep the buffer size to the sequence length required by the model
        if len(event_buffer) > SEQUENCE_LENGTH:
            event_buffer.pop(0) # Remove the oldest event

        if len(event_buffer) == SEQUENCE_LENGTH:
            # Prepare the request payload
            request_payload = {
                "events": [{"event": ev} for ev in event_buffer]
            }
            
            try:
                response = requests.post(API_URL, json=request_payload)
                response.raise_for_status() # Raise an exception for HTTP errors
                
                prediction_result = response.json()
                predictions.append({
                    "timestamp_seconds": event_data.get('timestamp_seconds'),
                    "win_prob": prediction_result.get('win_probability'),
                    "draw_prob": prediction_result.get('draw_probability'),
                    "loss_prob": prediction_result.get('loss_probability'),
                    "explanation": prediction_result.get('explanation')
                })
                
                print(f"Event {index+1} at {event_data.get('timestamp_seconds', 'N/A')}s - "
                      f"WP: {prediction_result['win_probability']:.2f}, "
                      f"DP: {prediction_result['draw_probability']:.2f}, "
                      f"LP: {prediction_result['loss_probability']:.2f}")
                
            except requests.exceptions.RequestException as e:
                print(f"Error sending event {index+1} to API: {e}")
                print(f"Response content: {response.text if response else 'N/A'}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON response for event {index+1}. Response: {response.text}")
            
        time.sleep(delay_seconds) # Simulate real-time delay

    print(f"Simulation complete for match: {match_event_file}")
    return predictions

if __name__ == "__main__":
    # Example usage:
    # Ensure your FastAPI server is running at http://localhost:8000
    
    # Find a sample processed event file
    sample_file = None
    if os.path.exists(PROCESSED_EVENTS_DIR):
        csv_files = [f for f in os.listdir(PROCESSED_EVENTS_DIR) if f.endswith('.csv')]
        if csv_files:
            sample_file = os.path.join(PROCESSED_EVENTS_DIR, csv_files)
            print(f"Using sample event file: {sample_file}")
        else:
            print(f"No CSV files found in {PROCESSED_EVENTS_DIR}. Please run main.py to preprocess data.")
    else:
        print(f"Directory {PROCESSED_EVENTS_DIR} not found. Please run main.py to preprocess data.")

    if sample_file:
        all_predictions = simulate_event_stream(sample_file, delay_seconds=0.05)
        if all_predictions:
            print("\n--- Sample of collected predictions ---")
            for p in all_predictions[:5]: # Print first 5 predictions
                print(p)