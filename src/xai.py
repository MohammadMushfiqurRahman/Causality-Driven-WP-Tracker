import shap
import numpy as np
import pandas as pd
import tensorflow as tf

def get_shap_values(model, background_data, data_instance):
    """
    Generates SHAP values for a given model prediction using a representative background dataset.
    
    Args:
        model: The trained Keras model.
        background_data: A representative sample of the training data (numpy array) to be used as the background for the SHAP explainer.
                         Shape should be (num_samples, sequence_length, num_features).
        data_instance: A single preprocessed event sequence (numpy array) for which to explain the prediction.
                       Shape should be (1, sequence_length, num_features).
                       
    Returns:
        shap_values: SHAP values for each feature.
        expected_value: The expected value of the model's output.
    """
    # Create a SHAP explainer using the provided background data
    explainer = shap.DeepExplainer(model, background_data)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(data_instance)
    
    # shap_values will be a list of arrays, one for each output class (Win, Draw, Loss)
    # We'll return the SHAP values for all classes.
    
    # The expected value is also returned by the explainer
    expected_value = explainer.expected_value
    
    return shap_values, expected_value

def explain_prediction(model, preprocessor, event_sequence, feature_names, top_n=3):
    """
    Explains a single model prediction using SHAP, returning top contributing features.
    
    Args:
        model: The trained Keras model.
        preprocessor: The StandardScaler used for feature scaling.
        event_sequence: A single preprocessed event sequence (numpy array) for which to explain the prediction.
                        Shape should be (sequence_length, num_features).
        feature_names: List of feature names.
        top_n: Number of top features to return.
        
    Returns:
        A dictionary containing:
        - 'prediction': The model's probability prediction.
        - 'explanation': A list of dictionaries, each with 'feature', 'value', 'shap_value'.
    """
    # Reshape for model prediction and SHAP explanation (add batch dimension)
    input_data = np.expand_dims(event_sequence, axis=0)
    
    # Get model prediction
    prediction = model.predict(input_data) # Get probabilities for the single instance
    
    # Get SHAP values
    shap_values_per_class, expected_value_per_class = get_shap_values(model, preprocessor, input_data, feature_names)
    
    # For simplicity, let's focus on explaining the predicted class (highest probability)
    predicted_class_index = np.argmax(prediction)
    shap_values_for_predicted_class = shap_values_per_class[predicted_class_index]
    
    # Average SHAP values across the sequence length for each feature
    # This gives an overall importance for each feature across the sequence
    avg_shap_values = np.mean(np.abs(shap_values_for_predicted_class), axis=0)
    
    # Create a DataFrame for easier sorting and selection
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_value': avg_shap_values
    })
    
    # Sort by absolute SHAP value to get top features
    feature_importance = feature_importance.sort_values(by='shap_value', ascending=False)
    
    explanation = []
    for _, row in feature_importance.head(top_n).iterrows():
        # To get the actual feature value, we need to inverse transform or get from original data
        # For now, we'll just show the feature name and its SHAP contribution
        explanation.append({
            'feature': row['feature'],
            'shap_value': row['shap_value']
        })
            
    return {
        'prediction': prediction.tolist(),
        'explanation': explanation
    }

if __name__ == "__main__":
    # Dummy example for demonstration
    # In a real scenario, you would load your trained model, scaler, and real data.
    
    # 1. Create a dummy model (similar to src/model.py)
    input_shape = (10, 3) # sequence_length, num_features
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Masking(mask_value=0.),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    dummy_model.compile(optimizer='adam', loss='categorical_crossentropy')

    # 2. Create a dummy preprocessor (StandardScaler)
    from sklearn.preprocessing import StandardScaler
    dummy_scaler = StandardScaler()
    
    # 3. Create dummy event sequence data
    dummy_event_sequence = np.random.rand(10, 3) # 10 events, 3 features
    dummy_feature_names = ['x_coordinate', 'y_coordinate', 'time_since_last_event']
    
    # 4. Explain a prediction
    explanation_result = explain_prediction(
        model=dummy_model,
        preprocessor=dummy_scaler,
        event_sequence=dummy_event_sequence,
        feature_names=dummy_feature_names,
        top_n=2
    )
    
    print("Explanation Result:")
    print(f"Prediction Probabilities: {explanation_result['prediction']}")
    print("Top Features Contributing to Prediction:")
    for item in explanation_result['explanation']:
        print(f"  - Feature: {item['feature']}, SHAP Value: {item['shap_value']:.4f}")