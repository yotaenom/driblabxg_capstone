#!/usr/bin/env python3
import sys
import argparse
import yaml
import os
import pickle
import pandas as pd
from datetime import datetime

# Try to import from both possible locations
import sys
import os
OLD_PREPROCESSING = False
NEW_PREPROCESSING = False

# Add current directory to Python path to ensure local imports work
if '.' not in sys.path:
    sys.path.insert(0, '.')

try:
    from src.data_preprocessing import preprocess_all_data as preprocess_all_data_old
    OLD_PREPROCESSING = True
    print("✓ Found src.data_preprocessing (original)")
    
    # Try to import prepare_features_for_inference, but don't fail if it doesn't exist
    try:
        from src.data_preprocessing import prepare_features_for_inference as prepare_features_old
        HAS_PREPARE_FEATURES = True
    except ImportError:
        HAS_PREPARE_FEATURES = False
        print("  - prepare_features_for_inference not found (will use fallback)")
        
except ImportError as e:
    print(f"✗ src.data_preprocessing not available: {e}")

try:
    import data_preprocessing
    from data_preprocessing import preprocess_all_data as preprocess_all_data_new
    NEW_PREPROCESSING = True
    print("✓ Found data_preprocessing (new)")
except ImportError as e:
    print(f"✗ data_preprocessing not available: {e}")

try:
    from src.data_loader import load_all_tracking_data, load_mappings, load_all_shots
    from src.data_validation import validate_all_data, validate_data_types
    from src.inference import run_inference_pipeline, generate_inference_report
    from src.logger import setup_logger
    USE_EXISTING_MODULES = True
    print("✓ Found existing src modules")
except ImportError as e:
    USE_EXISTING_MODULES = False
    print(f"✗ src modules not available: {e}")


def load_model(model_path: str):
    """Load the trained xG prediction model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model prediction."""
    model_features = [
        'R', 'shooting_angle', 'PDI_complete',
        'gk_line_offset'
    ]
    
    available_features = [col for col in model_features if col in df.columns]
    missing_features = [col for col in model_features if col not in df.columns]
    
    if missing_features:
        print(f"⚠ Warning: Missing features for prediction: {missing_features}")
    
    if not available_features:
        raise ValueError("No model features available in the data")
    
    feature_df = df[available_features].copy()
    
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val)
    
    return feature_df


def make_predictions(model, feature_df: pd.DataFrame) -> pd.Series:
    """Make xG predictions using the loaded model."""
    try:
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(feature_df)[:, 1]
        else:
            predictions = model.predict(feature_df)
        
        return pd.Series(predictions, index=feature_df.index)
        
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")


def save_results(df: pd.DataFrame, predictions: pd.Series, output_path: str):
    """Save the results to output directory."""
    os.makedirs(output_path, exist_ok=True)
    
    results_df = df.copy()
    results_df['xG_prediction'] = predictions
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    full_output_file = os.path.join(output_path, f"xg_predictions_full_{timestamp}.csv")
    results_df.to_csv(full_output_file, index=False)
    
    summary_columns = [
        'id', 'matchId', 'videoTimestamp', 'location_x', 'location_y',
        'team_id', 'player_id', 'xG_prediction'
    ]
    available_summary_cols = [col for col in summary_columns if col in results_df.columns]
    
    summary_df = results_df[available_summary_cols]
    summary_output_file = os.path.join(output_path, f"xg_predictions_summary_{timestamp}.csv")
    summary_df.to_csv(summary_output_file, index=False)
    
    print(f"Results saved to: {full_output_file}")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {}
    except yaml.YAMLError:
        return {}


def get_paths(config: dict, args) -> dict:
    """Get paths from config and override with command line arguments if provided."""
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    output_config = config.get('output', {})
    
    paths = {
        'shots_path': os.path.join(data_config.get('base_path', 'data/shot_pack'), 
                                  data_config.get('shots_dir', 'shots')),
        'tracking_path': os.path.join(data_config.get('base_path', 'data/shot_pack'), 
                                     data_config.get('tracking_dir', 'jsonls')),
        'mapping_path': os.path.join(data_config.get('base_path', 'data/shot_pack'), 
                                    data_config.get('mappings_dir', 'mappings')),
        'model_path': model_config.get('path', 'models'),
        'output_path': output_config.get('dir', 'output')
    }
    
    if args.shots_path:
        paths['shots_path'] = args.shots_path
    if args.tracking_path:
        paths['tracking_path'] = args.tracking_path
    if args.mapping_path:
        paths['mapping_path'] = args.mapping_path
    if args.model_path:
        paths['model_path'] = args.model_path
    if args.output_path:
        paths['output_path'] = args.output_path
    
    return paths


def main(args):
    if USE_EXISTING_MODULES and OLD_PREPROCESSING:
        # Use existing pipeline with original preprocessing
        logger = setup_logger()
        logger.info("Starting Driblab xG Prediction Pipeline")
        
        config = load_config(args.config)
        paths = get_paths(config, args)
        
        try:
            logger.info("Step 1/5: Loading data...")
            tracking = load_all_tracking_data(paths['tracking_path'], max_files=5)  # TEMP: Only load 5 files
            mappings = load_mappings(paths['mapping_path'])
            shots = load_all_shots(paths['shots_path'], max_files=5)  # TEMP: Only load 5 files
            data = {'tracking': tracking, 'mappings': mappings, 'shots': shots}

            logger.info("Step 2/5: Validating data...")
            validate_all_data(
                tracking_data=data['tracking'],
                mappings=data['mappings'],
                shots_data=data['shots']
            )
            validate_data_types(
                tracking_data=data['tracking'],
                mappings=data['mappings'],
                shots_data=data['shots']
            )

            logger.info("Step 3/5: Preprocessing data...")
            processed_df = preprocess_all_data_old(
                shots_dir=paths['shots_path'],
                tracking_dir=paths['tracking_path'],
                mappings_dir=paths['mapping_path'],
                max_files=5  # TEMP: Only process 5 files for quick test
            )
            
            # Use prepare_features_for_inference if available, otherwise skip
            if HAS_PREPARE_FEATURES:
                features_df = prepare_features_old(processed_df)
            else:
                features_df = processed_df
                print("  - Using processed data directly (no prepare_features_for_inference)")

            logger.info("Step 4/5: Running inference...")
            # Pass the shots_dir parameter to extract missing features
            results = run_inference_pipeline(
                features_df,
                model_path=paths['model_path'],
                output_dir=paths['output_path'],
                shots_dir=paths['shots_path']  # Add this parameter
            )

            logger.info("Step 5/5: Generating report...")
            report = generate_inference_report(results)
            print(report)

            logger.info("Pipeline completed successfully!")
            print(f"Results saved to: {results['output_file']}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"\nPipeline failed: {e}")
            sys.exit(1)
    
    elif NEW_PREPROCESSING:
        # Use new pipeline with tracking integration
        config = load_config(args.config)
        paths = get_paths(config, args)
        
        for path_name, path_value in [
            ('shots_path', paths['shots_path']),
            ('tracking_path', paths['tracking_path']),
            ('mapping_path', paths['mapping_path'])
        ]:
            if not os.path.exists(path_value):
                raise FileNotFoundError(f"{path_name} not found: {path_value}")
        
        if not os.path.exists(paths['model_path']):
            raise FileNotFoundError(f"Model file not found: {paths['model_path']}")
        
        try:
            print("Step 1: Loading and preprocessing data...")
            processed_df = preprocess_all_data_new(
                shots_dir=paths['shots_path'],
                tracking_dir=paths['tracking_path'],
                mappings_dir=paths['mapping_path'],
                prev_frames=10,
                next_frames=10,
                max_files=5  # TEMP: Only process 5 files for quick test
            )
            
            if processed_df.empty:
                raise ValueError("No data available after preprocessing")
            
            print("Step 2: Loading prediction model...")
            model = load_model(paths['model_path'])
            
            print("Step 3: Preparing features for prediction...")
            feature_df = prepare_features_for_prediction(processed_df)
            
            print("Step 4: Making xG predictions...")
            predictions = make_predictions(model, feature_df)
            
            print("Step 5: Saving results...")
            save_results(processed_df, predictions, paths['output_path'])
            
            print("✓ xG Prediction Pipeline Completed Successfully!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Pipeline failed. Check the error message above.")
            return 1
    
    else:
        print("❌ Error: No preprocessing modules available")
        print("Please ensure either:")
        print("1. src/data_preprocessing.py exists with your original functions, OR")
        print("2. data_preprocessing.py exists with the new tracking integration")
        return 1
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driblab xG Prediction Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--shots_path", help="Path to shots directory")
    parser.add_argument("--tracking_path", help="Path to tracking directory")
    parser.add_argument("--mapping_path", help="Path to mappings directory")
    parser.add_argument("--model_path", help="Path to model file")
    parser.add_argument("--output_path", help="Path to output directory")
    args = parser.parse_args()
    main(args)