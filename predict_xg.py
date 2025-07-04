import sys
import argparse
import yaml
import os
from src.data_loader import load_all_tracking_data, load_mappings, load_all_shots
from src.data_validation import validate_all_data, validate_data_types
from src.data_preprocessing import preprocess_all_data, prepare_features_for_inference
from src.inference import run_inference_pipeline, generate_inference_report
from src.logger import setup_logger


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}


def get_paths(config: dict, args) -> dict:
    """
    Get paths from config and override with command line arguments if provided.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Dictionary of paths
    """
    # Default paths from config
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
    
    # Override with command line arguments if provided
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
    # Setup logger
    logger = setup_logger()
    logger.info("Starting Driblab xG Prediction Pipeline")
    
    # Load configuration
    config = load_config(args.config)
    paths = get_paths(config, args)
    
    logger.info(f"Using configuration: {paths}")
    
    try:
        # Step 1: Load data
        logger.info("Step 1/5: Loading data...")
        tracking = load_all_tracking_data(paths['tracking_path'])
        mappings = load_mappings(paths['mapping_path'])
        shots = load_all_shots(paths['shots_path'])
        data = {'tracking': tracking, 'mappings': mappings, 'shots': shots}

        # Step 2: Validate data
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

        # Step 3: Preprocess data
        logger.info("Step 3/5: Preprocessing data...")
        processed_df = preprocess_all_data(
            tracking_data=data['tracking'],
            mappings=data['mappings'],
            shots_data=data['shots']
        )
        features_df = prepare_features_for_inference(processed_df)

        # Step 4: Run inference
        logger.info("Step 4/5: Running inference...")
        results = run_inference_pipeline(
            features_df,
            model_path=paths['model_path'],
            output_dir=paths['output_path']
        )

        # Step 5: Output report
        logger.info("Step 5/5: Generating report...")
        report = generate_inference_report(results)
        print(report)

        logger.info("Pipeline completed successfully!")
        print(f"Results saved to: {results['output_file']}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driblab xG Prediction Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--shots_path", help="Path to shots directory (overrides config)")
    parser.add_argument("--tracking_path", help="Path to tracking directory (overrides config)")
    parser.add_argument("--mapping_path", help="Path to mappings directory (overrides config)")
    parser.add_argument("--model_path", help="Path to model file or directory (overrides config)")
    parser.add_argument("--output_path", help="Path to output directory (overrides config)")
    args = parser.parse_args()
    main(args)
