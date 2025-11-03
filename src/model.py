import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

class WinProbabilityModel:
    def __init__(self, input_shape, num_teams=2, num_outputs=3, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_teams = num_teams
        self.num_outputs = num_outputs  # Win, Draw, Loss probabilities
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Input layer for sequential event data
        event_input = Input(shape=self.input_shape, name='event_sequence_input')
        
        # Masking layer to handle variable-length sequences (padding)
        masked_input = Masking(mask_value=0., name='masking_layer')(event_input)
        
        # LSTM layer to capture temporal dependencies
        lstm_out = LSTM(128, return_sequences=False, name='lstm_layer')(masked_input)
        
        # Dense layers for processing and output
        dense_1 = Dense(64, activation='relu', name='dense_1')(lstm_out)
        output = Dense(self.num_outputs, activation='softmax', name='output_layer')(dense_1)
        
        model = Model(inputs=event_input, outputs=output)
        
        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        print("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        print("Model training complete.")
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path="models/win_probability_model.h5"):
        self.model.save(path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path="models/win_probability_model.h5"):
        model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        # Create a dummy instance to hold the loaded model
        instance = cls(input_shape=(1,1)) # input_shape is dummy here, actual model is loaded, will be overwritten
        instance.model = model
        return instance

def prepare_data_for_model(processed_events_dir="data/processed_events", sequence_length=10):
    """
    Loads processed event data, creates sequences, and prepares for model training.
    This is a simplified example. Real feature engineering would be more complex.
    """
    all_matches_df = []
    for filename in os.listdir(processed_events_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(processed_events_dir, filename)
            df = pd.read_csv(file_path)
            all_matches_df.append(df)
    
    if not all_matches_df:
        print("No processed event data found. Please run data ingestion first.")
        return None, None, None, None

    full_df = pd.concat(all_matches_df, ignore_index=True)

    # --- Simplified Feature Engineering ---
    # For demonstration, let's use 'x' and 'y' coordinates and 'timestamp_seconds' as features.
    # In a real scenario, you'd have many more engineered features.
    features = ['timestamp_seconds', 'pitch_control_score', 'time_since_last_event', 'distance_to_goal']
    
    # Filter for events that have these features
    feature_df = full_df.dropna(subset=features)

    if feature_df.empty:
        print("No valid features found after dropping NaNs.")
        return None, None, None, None

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df[features])
    scaled_feature_df = pd.DataFrame(scaled_features, columns=features, index=feature_df.index)

    # Create sequences
    X, y = [], []
    # For simplicity, let's assume a target based on the final score of a match.
    # This needs to be properly linked to match outcomes.
    # For now, we'll create dummy targets.
    
    # Dummy target: 0 for Loss, 1 for Draw, 2 for Win
    # In a real scenario, you'd determine the match outcome for each sequence.
    dummy_targets = np.random.randint(0, 3, len(scaled_feature_df)) 
    
    for i in range(len(scaled_feature_df) - sequence_length):
        X.append(scaled_feature_df.iloc[i:i+sequence_length].values)
        y.append(dummy_targets[i+sequence_length-1]) # Predict outcome after sequence

    X = np.array(X)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=3) # One-hot encode

    # Pad sequences to ensure consistent length (if not already handled by Masking layer)
    # For simplicity, we assume fixed sequence_length here.
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, scaler, features # Return scaler and feature names for later use in inference and XAI

if __name__ == "__main__":
    # Example usage
    sequence_length = 10
    X_train, X_val, y_train, y_val, scaler = prepare_data_for_model(sequence_length=sequence_length)

    if X_train is not None:
        input_shape = (X_train.shape[1], X_train.shape[2]) # (sequence_length, num_features)
        model_instance = WinProbabilityModel(input_shape=input_shape)
        model_instance.model.summary()

        # Train the model
        model_instance.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64)

        # Save the trained model
        model_instance.save_model()

        # Load the model (example)
        loaded_model_instance = WinProbabilityModel.load_model()
        
        # Make predictions (example)
        sample_prediction = loaded_model_instance.predict(X_val[:1])
        print(f"Sample prediction: {sample_prediction}")