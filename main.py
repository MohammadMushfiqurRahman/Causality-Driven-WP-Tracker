import os
from src.data_ingestion import download_statsbomb_data, load_and_preprocess_events
from src.feature_engineering import run_feature_engineering
from src.model import WinProbabilityModel, prepare_data_for_model
from src.xai import explain_prediction

def main():
    # Define competition and season IDs
    competition_id = 11  # Example: La Liga
    season_id = 27       # Example: 2015/2016 Season

    raw_events_dir = "data/raw_events"
    processed_events_dir = "data/processed_events"
    engineered_features_dir = "data/engineered_features"
    models_dir = "models"

    # 1. Download StatsBomb data (only if raw_events_dir is empty)
    if not os.path.exists(raw_events_dir) or not os.listdir(raw_events_dir):
        print("Starting data download...")
        download_statsbomb_data(competition_id, season_id, raw_events_dir)
        print("Data download complete.")
    else:
        print("Raw event data already exists. Skipping download.")

    # 2. Preprocess downloaded data (only if processed_events_dir is empty)
    if not os.path.exists(processed_events_dir) or not os.listdir(processed_events_dir):
        print("Starting data preprocessing...")
        if not os.path.exists(processed_events_dir):
            os.makedirs(processed_events_dir)

        for filename in os.listdir(raw_events_dir):
            if filename.startswith("events_") and filename.endswith(".json"):
                file_path = os.path.join(raw_events_dir, filename)
                print(f"Processing {filename}...")
                df = load_and_preprocess_events(file_path)
                
                # Save processed data (e.g., to CSV or Parquet)
                output_path = os.path.join(processed_events_dir, filename.replace(".json", ".csv"))
                df.to_csv(output_path, index=False)
                print(f"Saved processed data to {output_path}")
        print("Data preprocessing complete.")
    else:
        print("Processed event data already exists. Skipping preprocessing.")

    # 3. Run Feature Engineering
    print("Starting feature engineering...")
    run_feature_engineering(processed_events_dir, engineered_features_dir)
    print("Feature engineering complete.")

    # 4. Prepare data for model and train
    print("Preparing data for model training...")
    sequence_length = 10 # Define your desired sequence length
    X_train, X_val, y_train, y_val, scaler, feature_names = prepare_data_for_model(
        processed_events_dir=engineered_features_dir, # Use engineered features
        sequence_length=sequence_length
    )

    if X_train is not None:
        input_shape = (X_train.shape, X_train.shape) # (sequence_length, num_features)
        model_instance = WinProbabilityModel(input_shape=input_shape)
        model_instance.model.summary()

        # Train the model
        model_instance.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64)

        # Save the trained model
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_instance.save_model(os.path.join(models_dir, "win_probability_model.h5"))

        # Example of XAI integration
        print("\nGenerating SHAP explanations for a sample prediction...")
        # Assuming X_val is a representative sample
        sample_event_sequence = X_val # Take the first sequence from validation set
        
        explanation_result = explain_prediction(
            model=model_instance.model,
            preprocessor=scaler, # Use the scaler fitted during data preparation
            event_sequence=sample_event_sequence,
            feature_names=feature_names,
            top_n=3
        )
        print("Explanation Result:")
        print(f"Prediction Probabilities: {explanation_result['prediction']}")
        print("Top Features Contributing to Prediction:")
        for item in explanation_result['explanation']:
            print(f"  - Feature: {item['feature']}, SHAP Value: {item['shap_value']:.4f}")

    else:
        print("Model training skipped due to insufficient data.")

if __name__ == "__main__":
    main()