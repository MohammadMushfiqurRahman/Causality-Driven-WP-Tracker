import streamlit as st
import pandas as pd
import numpy as np
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

def run_simulation(match_file, chart_placeholder, info_placeholder, leverage_placeholder, event_log_placeholder):
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
    event_log = []
    last_win_prob = None
    last_draw_prob = None
    last_loss_prob = None
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
                explanation = prediction.get('explanation', [])

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

                # --- Leverage & Event Log Calculation (F.R. 2.2 & 2.3) ---
                leverage_score = 0
                wp_change = 0
                if last_win_prob is not None:
                    # Leverage is the absolute change in Win Probability
                    leverage_score = abs(win_prob - last_win_prob)
                    wp_change = win_prob - last_win_prob

                event_log.append({
                    'timestamp': event_data.get('timestamp_seconds'),
                    'event_type': event_data.get('type_name', 'Unknown Event'),
                    'player': event_data.get('player_name', 'N/A'),
                    'win_prob': win_prob,
                    'draw_prob': draw_prob,
                    'loss_prob': loss_prob,
                    'leverage_score': leverage_score,
                    'wp_change': wp_change,
                    'explanation': explanation
                })

                # Update the last known probabilities
                last_win_prob = win_prob
                last_draw_prob = draw_prob
                last_loss_prob = loss_prob


            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                st.error(f"API Error at event {index+1}: {e}")
                break

        time.sleep(0.05)  # Simulate delay

    st.success("Simulation complete.")

    # --- Display Top 5 Leverage Moments (F.R. 2.2) ---
    if event_log:
        leverage_placeholder.header("Top 5 Moments of Maximum Leverage")
        # Sort events by leverage score, descending
        top_leverage_events = sorted(event_log, key=lambda x: x['leverage_score'], reverse=True)[:5]

        for i, event in enumerate(top_leverage_events):
            wp_change_str = f"+{event['wp_change']:.1%}" if event['wp_change'] > 0 else f"{event['wp_change']:.1%}"
            expander_title = (
                f"**{i+1}. {event['event_type']} by {event['player']} at {event['timestamp']:.2f}s** "
                f"(WP Change: {wp_change_str})"
            )
            with leverage_placeholder.expander(expander_title):
                st.write("**Causal Explanation (Top Features):**")
                for feature_info in event['explanation']:
                    st.markdown(f"- **{feature_info['feature']}**: `SHAP Value: {feature_info['shap_value']:.4f}`")

    # --- Display Event Causality Breakdown (F.R. 2.3) ---
    if event_log:
        event_log_placeholder.header("Event Causality Breakdown")
        log_df = pd.DataFrame(event_log)

        # --- Filtering UI ---
        col1, col2, col3 = event_log_placeholder.columns(3)

        # Player Filter
        unique_players = sorted(log_df['player'].dropna().unique())
        selected_players = col1.multiselect("Filter by Player:", unique_players)

        # Event Type Filter
        unique_event_types = sorted(log_df['event_type'].dropna().unique())
        selected_event_types = col2.multiselect("Filter by Event Type:", unique_event_types)

        # Leverage Score Filter
        min_leverage = col3.slider(
            "Minimum Leverage Score:",
            min_value=0.0,
            max_value=log_df['leverage_score'].max() if not log_df.empty else 1.0,
            value=0.0,
            step=0.01
        )

        # --- Apply Filters ---
        filtered_df = log_df.copy()
        if selected_players:
            filtered_df = filtered_df[filtered_df['player'].isin(selected_players)]
        if selected_event_types:
            filtered_df = filtered_df[filtered_df['event_type'].isin(selected_event_types)]
        if min_leverage > 0.0:
            filtered_df = filtered_df[filtered_df['leverage_score'] >= min_leverage]

        # --- Display Filtered Log ---
        display_cols = ['timestamp', 'event_type', 'player', 'win_prob', 'draw_prob', 'loss_prob', 'leverage_score']
        event_log_placeholder.dataframe(
            filtered_df[display_cols].style.format({'timestamp': '{:.2f}', 'win_prob': '{:.3f}', 'draw_prob': '{:.3f}', 'loss_prob': '{:.3f}', 'leverage_score': '{:.4f}'}),
            use_container_width=True
        )

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
        leverage_placeholder = st.container()
        event_log_placeholder = st.container()
        run_simulation(selected_match, chart_placeholder, info_placeholder, leverage_placeholder, event_log_placeholder)
    else:
        st.info("Select a match and click 'Start Simulation' to begin.")