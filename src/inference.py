import os
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime


def load_model(model_path: str = "models/") -> Any:
    """
    Loads the trained model from the models folder.
    
    Args:
        model_path: Path to the models directory or specific model file
        
    Returns:
        Loaded model object
    """
    if os.path.isdir(model_path):
        # Look for model files in the directory
        model_files = [f for f in os.listdir(model_path) 
                      if f.endswith(('.pkl', '.joblib', '.pickle', '.h5', '.hdf5'))]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_path}")
        
        # Use the first model file found (or implement logic to choose specific one)
        model_file = model_files[0]
        full_path = os.path.join(model_path, model_file)
    else:
        full_path = model_path
    
    try:
        # Try loading with joblib first (for sklearn models)
        if full_path.endswith(('.pkl', '.joblib', '.pickle')):
            model = joblib.load(full_path)
        # TODO: Add support for other model formats
        # elif full_path.endswith(('.h5', '.hdf5')):
        #     from tensorflow import keras
        #     model = keras.models.load_model(full_path)
        else:
            raise ValueError(f"Unsupported model format: {full_path}")
        
        print(f"✓ Model loaded successfully from: {full_path}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {full_path}: {e}")


def prepare_features_for_model(features_df: pd.DataFrame) -> np.ndarray:
    """
    Prepares the features DataFrame for model prediction.
    
    Args:
        features_df: DataFrame with features from preprocessing
        
    Returns:
        numpy array ready for model prediction
    """
    # Select only the features that the model expects
    model_features = [
        'R', 'shooting_angle', 'PDI_complete',
        'gk_line_offset'
    ]
    
    available_features = [col for col in model_features if col in features_df.columns]
    missing_features = [col for col in model_features if col not in features_df.columns]
    
    if missing_features:
        print(f"⚠ Warning: Missing features for prediction: {missing_features}")
    
    if not available_features:
        raise ValueError("No model features available in the data")
    
    # Select only the available model features
    feature_df = features_df[available_features].copy()
    
    # Handle missing values
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val)
    
    # Convert to numpy array
    features_array = feature_df.values
    
    print(f"✓ Features prepared for model: {features_array.shape}")
    return features_array


def run_inference(model: Any, features: np.ndarray) -> np.ndarray:
    """
    Runs model inference on the prepared features.
    
    Args:
        model: Loaded model object
        features: Prepared features as numpy array
        
    Returns:
        Model predictions
    """
    try:
        # TODO: Add model-specific prediction logic
        # Examples:
        # - model.predict() for sklearn models
        # - model.predict_proba() for probability predictions
        # - model() for tensorflow/pytorch models
        
        # Placeholder: Assume sklearn-style predict method
        predictions = model.predict(features)
        
        print(f"✓ Inference completed: {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")


def create_results_dataframe(features_df: pd.DataFrame, 
                           predictions: np.ndarray,
                           prediction_column: str = "predicted_xg") -> pd.DataFrame:
    """
    Creates a results DataFrame with original features and predictions.
    
    Args:
        features_df: Original features DataFrame
        predictions: Model predictions
        prediction_column: Name for the prediction column
        
    Returns:
        DataFrame with features and predictions
    """
    results_df = features_df.copy()
    results_df[prediction_column] = predictions
    
    print(f"✓ Results DataFrame created: {results_df.shape}")
    return results_df


def save_results(results_df: pd.DataFrame, 
                output_dir: str = "output/",
                filename: Optional[str] = None) -> str:
    """
    Saves the results to the output folder.
    
    Args:
        results_df: DataFrame with results
        output_dir: Directory to save results
        filename: Optional custom filename
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xg_predictions_{timestamp}.csv"
    
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        results_df.to_csv(filepath, index=False)
        print(f"✓ Results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        raise RuntimeError(f"Failed to save results to {filepath}: {e}")


def run_inference_pipeline(features_df: pd.DataFrame,
                          model_path: str = "models/",
                          output_dir: str = "output/",
                          prediction_column: str = "predicted_xg") -> Dict[str, Any]:
    """
    Complete inference pipeline: load model, run predictions, save results.
    
    Args:
        features_df: Preprocessed features DataFrame
        model_path: Path to model file or directory
        output_dir: Directory to save results
        prediction_column: Name for the prediction column
        
    Returns:
        Dictionary with results and metadata
    """
    print("Starting inference pipeline...")
    
    # Step 1: Load model
    model = load_model(model_path)
    
    # Step 2: Prepare features for model
    features = prepare_features_for_model(features_df)
    
    # Step 3: Run inference
    predictions = run_inference(model, features)
    
    # Step 4: Create results DataFrame
    results_df = create_results_dataframe(features_df, predictions, prediction_column)
    
    # Step 5: Save results
    output_file = save_results(results_df, output_dir)
    
    # Step 6: Return results and metadata
    results = {
        'predictions': predictions,
        'results_df': results_df,
        'output_file': output_file,
        'model_path': model_path,
        'num_predictions': len(predictions),
        'feature_shape': features.shape
    }
    
    print("✓ Inference pipeline completed successfully!")
    return results


def generate_inference_report(results: Dict[str, Any]) -> str:
    """
    Generates a simple inference report.
    
    Args:
        results: Results dictionary from inference pipeline
        
    Returns:
        Report string
    """
    report = f"""
Inference Report
================
Model Path: {results['model_path']}
Number of Predictions: {results['num_predictions']}
Feature Shape: {results['feature_shape']}
Output File: {results['output_file']}

Prediction Statistics:
- Min: {np.min(results['predictions']):.4f}
- Max: {np.max(results['predictions']):.4f}
- Mean: {np.mean(results['predictions']):.4f}
- Std: {np.std(results['predictions']):.4f}
"""
    return report


if __name__ == "__main__":
    # Test the inference pipeline
    from src.data_loader import load_all_data
    from src.data_preprocessing import preprocess_all_data, prepare_features_for_inference
    
    try:
        # Load and preprocess data
        data = load_all_data()
        processed_df = preprocess_all_data(
            tracking_data=data['tracking'],
            mappings=data['mappings'],
            shots_data=data['shots']
        )
        features_df = prepare_features_for_inference(processed_df)
        
        # Run inference pipeline
        results = run_inference_pipeline(features_df)
        
        # Generate and print report
        report = generate_inference_report(results)
        print(report)
        
    except Exception as e:
        print(f"Inference pipeline failed: {e}")
