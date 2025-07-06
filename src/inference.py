import os
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer


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


def prepare_original_shot_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the original shot event features that are missing from the tracking pipeline.
    These features need to be extracted from the original shot data.
    
    Args:
        features_df: DataFrame with features from preprocessing
        
    Returns:
        DataFrame with additional original shot features
    """
    df = features_df.copy()
    
    # Initialize missing features with default values
    # These would normally come from the original shot event data
    
    # Binary features that need to be extracted from shot event data
    if 'assist_type_through_pass' not in df.columns:
        df['assist_type_through_pass'] = 0
        print("⚠ Warning: assist_type_through_pass not available, using default value 0")
    
    if 'assist_type_long_pass' not in df.columns:
        df['assist_type_long_pass'] = 0
        print("⚠ Warning: assist_type_long_pass not available, using default value 0")
    
    if 'shot_bodyPart_head_or_other' not in df.columns:
        df['shot_bodyPart_head_or_other'] = 0
        print("⚠ Warning: shot_bodyPart_head_or_other not available, using default value 0")
    
    if 'direct_free_kick' not in df.columns:
        df['direct_free_kick'] = 0
        print("⚠ Warning: direct_free_kick not available, using default value 0")
    
    return df


def extract_shot_features_from_raw_data(shots_dir: str) -> pd.DataFrame:
    """
    Extract the original shot event features directly from the raw shot files.
    This should be called before or integrated with the tracking preprocessing.
    
    Args:
        shots_dir: Path to the shots directory
        
    Returns:
        DataFrame with original shot features
    """
    import json
    
    shots_data = []
    for filename in os.listdir(shots_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(shots_dir, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    shots_data.extend(data)
                else:
                    shots_data.append(data)
    
    # Convert to DataFrame and normalize
    shots_df = pd.json_normalize(shots_data, sep='_')
    
    # Extract and process the original features
    
    # 1. One-hot encode shot_bodyPart
    if 'shot_bodyPart' in shots_df.columns:
        df_bodypart = pd.get_dummies(shots_df['shot_bodyPart'], prefix='shot_bodyPart')
        shots_df = pd.concat([shots_df, df_bodypart], axis=1)
        
        # Create shot_bodyPart_head_or_other
        head_cols = [col for col in df_bodypart.columns if 'head' in col.lower() or 'other' in col.lower()]
        if head_cols:
            shots_df['shot_bodyPart_head_or_other'] = shots_df[head_cols].max(axis=1).astype(int)
        else:
            shots_df['shot_bodyPart_head_or_other'] = 0
    else:
        shots_df['shot_bodyPart_head_or_other'] = 0
    
    # 2. Process assist_info_type_secondary
    if 'assist_info_type_secondary' in shots_df.columns:
        shots_df['assist_info_type_secondary'] = shots_df['assist_info_type_secondary'].fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        mlb_assist = MultiLabelBinarizer()
        assist_encoded = mlb_assist.fit_transform(shots_df['assist_info_type_secondary'])
        assist_df = pd.DataFrame(assist_encoded, 
                                columns=[f"assist_type_{col}" for col in mlb_assist.classes_],
                                index=shots_df.index)
        shots_df = pd.concat([shots_df, assist_df], axis=1)
        
        # Extract specific assist types
        long_pass_cols = [col for col in assist_df.columns if 'long_pass' in col.lower()]
        shots_df['assist_type_long_pass'] = (
            assist_df[long_pass_cols[0]].astype(int) if long_pass_cols 
            else 0
        )
        
        through_pass_cols = [col for col in assist_df.columns if 'through_pass' in col.lower()]
        shots_df['assist_type_through_pass'] = (
            assist_df[through_pass_cols[0]].astype(int) if through_pass_cols
            else 0
        )
    else:
        shots_df['assist_type_long_pass'] = 0
        shots_df['assist_type_through_pass'] = 0
    
    # 3. Process possession_types
    if 'possession_types' in shots_df.columns:
        shots_df['possession_types'] = shots_df['possession_types'].fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        mlb_possession = MultiLabelBinarizer()
        possession_encoded = mlb_possession.fit_transform(shots_df['possession_types'])
        possession_df = pd.DataFrame(possession_encoded,
                                    columns=mlb_possession.classes_,
                                    index=shots_df.index)
        shots_df = pd.concat([shots_df, possession_df], axis=1)
        
        # Extract direct_free_kick
        free_kick_cols = [col for col in possession_df.columns if 'direct_free_kick' in col.lower()]
        shots_df['direct_free_kick'] = (
            possession_df[free_kick_cols[0]].astype(int) if free_kick_cols
            else 0
        )
    else:
        shots_df['direct_free_kick'] = 0
    
    # Convert videoTimestamp to string for merging
    if 'videoTimestamp' in shots_df.columns:
        shots_df['videoTimestamp'] = shots_df['videoTimestamp'].astype(str)
    
    return shots_df


def prepare_features_for_model(features_df: pd.DataFrame, shots_dir: str = None) -> np.ndarray:
    """
    Prepares the features DataFrame for model prediction.
    Ensures exactly the 10 features expected by the model in the correct order.
    
    Args:
        features_df: DataFrame with features from preprocessing
        shots_dir: Optional path to shots directory to extract missing features
        
    Returns:
        numpy array ready for model prediction
    """
    # The exact features expected by the model in order
    expected_features = [
        "R",
        "theta", 
        "shooting_angle",
        "assist_type_through_pass",
        "assist_type_long_pass",
        "shot_bodyPart_head_or_other",
        "direct_free_kick",
        "PDI",
        "PDI_complete",
        "gk_line_offset"
    ]
    
    df = features_df.copy()
    
    # If we have access to the shots directory, extract the missing features
    if shots_dir and os.path.exists(shots_dir):
        print("Extracting original shot features from raw data...")
        shot_features_df = extract_shot_features_from_raw_data(shots_dir)
        
        # Merge on videoTimestamp
        if 'videoTimestamp' in df.columns and 'videoTimestamp' in shot_features_df.columns:
            # Select only the features we need from shot_features_df
            merge_features = ['videoTimestamp', 'assist_type_through_pass', 'assist_type_long_pass', 
                            'shot_bodyPart_head_or_other', 'direct_free_kick']
            available_merge_features = [col for col in merge_features if col in shot_features_df.columns]
            
            df = df.merge(shot_features_df[available_merge_features], on='videoTimestamp', how='left')
    
    # If features are still missing, add them with default values
    df = prepare_original_shot_features(df)
    
    # Check which features are available
    available_features = []
    missing_features = []
    
    for feature in expected_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
            # Add missing feature with default value
            df[feature] = 0
            print(f"⚠ Warning: {feature} missing, using default value 0")
    
    # Select features in the exact order expected by the model
    model_df = df[expected_features].copy()
    
    # Handle missing values
    for col in model_df.columns:
        if model_df[col].isna().any():
            if col in ['R', 'theta', 'shooting_angle', 'PDI', 'PDI_complete', 'gk_line_offset']:
                # For numeric features, use median
                median_val = model_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                model_df[col] = model_df[col].fillna(median_val)
            else:
                # For binary features, use 0
                model_df[col] = model_df[col].fillna(0)
    
    # Convert to numpy array
    features_array = model_df.values
    
    print(f"✓ Features prepared for model: {features_array.shape}")
    print(f"✓ Feature order: {expected_features}")
    print(f"✓ Available features: {len(available_features)}/{len(expected_features)}")
    if missing_features:
        print(f"⚠ Missing features (using defaults): {missing_features}")
    
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
        # Check if model has predict_proba method (for classification)
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(features)
            # If binary classification, return probabilities for positive class
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]
        else:
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
                          prediction_column: str = "predicted_xg",
                          shots_dir: str = None) -> Dict[str, Any]:
    """
    Complete inference pipeline: load model, run predictions, save results.
    
    Args:
        features_df: Preprocessed features DataFrame
        model_path: Path to model file or directory
        output_dir: Directory to save results
        prediction_column: Name for the prediction column
        shots_dir: Optional path to shots directory for extracting missing features
        
    Returns:
        Dictionary with results and metadata
    """
    print("Starting inference pipeline...")
    
    # Step 1: Load model
    model = load_model(model_path)
    
    # Step 2: Prepare features for model
    features = prepare_features_for_model(features_df, shots_dir)
    
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
    from src.data_preprocessing import preprocess_all_data
    
    try:
        # Load and preprocess data
        data = load_all_data()
        processed_df = preprocess_all_data(
            shots_dir="data/shot_pack/shots",
            tracking_dir="data/shot_pack/jsonls", 
            mappings_dir="data/shot_pack/mappings"
        )
        
        # Run inference pipeline
        results = run_inference_pipeline(
            processed_df,
            shots_dir="data/shot_pack/shots"  # Pass shots_dir to extract missing features
        )
        
        # Generate and print report
        report = generate_inference_report(results)
        print(report)
        
    except Exception as e:
        print(f"Inference pipeline failed: {e}")