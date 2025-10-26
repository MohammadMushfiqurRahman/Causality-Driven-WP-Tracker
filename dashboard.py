import streamlit as st
import pandas as pd
import requests
import time
import os
import plotly.graph_objects as go
import json

# --- Configuration ---
API_URL = "http://localhost:8000/predict"
PROCESSED_EVENTS_DIR = "data/processed_events"
SEQUENCE_LENGTH = 10  # Must match the model's expected sequence length

st.set_page_config(
    page_title="Win Probability Tracker",
    layout="wide"
)

st.title("Causality-Driven Win Probability Tracker")

def get_available_matches():
    """Gets a list of available match CSV files."""
    if not os.path.exists(PROCESSED_EVENTS_DIR):
        return []
    return [f for f in os.listdir(PROCESSED_EVENTS_DIR) if f.endswith('.csv')]

def run_simulation(match_file, chart_placeholder, info_placeholder):
    """Simulates a real-time event stream and updates the dashboard."""
    st.info(f"Starting simulation for: {match_file}")

    try:
        events_df = pd.read_csv(os.path.join(PROCESSED_EVENTS_DIR, match_file))
    except FileNotFoundError:
        st.error(f"Error: Event file not found at {match_file}")
        return

    # Sort events by timestamp
    if 'timestamp_seconds' in events_df.columns:
        events_df = events_df.sort_values(by='timestamp_seconds').reset_index(drop=True)
    else:
        st.warning("Warning: 'timestamp_seconds' column not found. Events may not be chronological.")

    # Initialize data for plotting
    plot_data = pd.DataFrame(columns=['time', 'win', 'draw', 'loss'])

    # Initialize Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Home Win', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Draw', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Away Win', line=dict(color='red')))
    fig.update_layout(
        title="Real-Time Win Probability",
        xaxis_title="Match Time (seconds)",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1])
    )

    event_buffer = []
    for index, event in events_df.iterrows():
        event_data = event.to_dict()
        event_buffer.append(event_data)

        if len(event_buffer) > SEQUENCE_LENGTH:
            event_buffer.pop(0)

        if len(event_buffer) == SEQUENCE_LENGTH:
            request_payload = {"events": [{"event": ev} for ev in event_buffer]}

            try:
                response = requests.post(API_URL, json=request_payload)
                response.raise_for_status()
                prediction = response.json()

                # The API now returns individual probabilities.
                win_prob = prediction.get('win_probability', 0)
                draw_prob = prediction.get('draw_probability', 0)
                loss_prob = prediction.get('loss_probability', 0)

                # Update plot data
                new_row = pd.DataFrame([{
                    'time': event_data.get('timestamp_seconds'),
                    'win': win_prob,
                    'draw': draw_prob,
                    'loss': loss_prob
                }])
                plot_data = pd.concat([plot_data, new_row], ignore_index=True)

                # Update chart
                fig.data[0].x = plot_data['time']
                fig.data[0].y = plot_data['win']
                fig.data[1].x = plot_data['time']
                fig.data[1].y = plot_data['draw']
                fig.data[2].x = plot_data['time']
                fig.data[2].y = plot_data['loss']
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                # Display latest prediction info
                info_placeholder.info(
                    f"Time: {event_data.get('timestamp_seconds', 'N/A'):.2f}s | "
                    f"Win: {win_prob:.2f} | Draw: {draw_prob:.2f} | Loss: {loss_prob:.2f}"
                )

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                st.error(f"API Error at event {index+1}: {e}")
                break

        time.sleep(0.05)  # Simulate delay

    st.success("Simulation complete.")

# --- Main UI ---
st.sidebar.header("Match Simulation")
available_matches = get_available_matches()

if not available_matches:
    st.sidebar.error(f"No processed event files found in `{PROCESSED_EVENTS_DIR}`. Please run `main.py` first.")
else:
    selected_match = st.sidebar.selectbox("Choose a match to simulate:", available_matches)

    if st.sidebar.button("Start Simulation"):
        # Placeholders for the chart and info text
        chart_placeholder = st.empty()
        info_placeholder = st.empty()
        run_simulation(selected_match, chart_placeholder, info_placeholder)
    else:
        st.info("Select a match and click 'Start Simulation' to begin.")