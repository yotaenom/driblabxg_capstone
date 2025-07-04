import sys
import argparse
from src.data_loader import load_all_tracking_data, load_mappings, load_all_shots
from src.data_validation import validate_all_data, validate_data_types
from src.data_preprocessing import preprocess_all_data, prepare_features_for_inference
from src.inference import run_inference_pipeline, generate_inference_report


def main(args):
    print("\n==== Driblab xG Prediction Pipeline ====")
    try:
        # Step 1: Load data
        print("\n[1/5] Loading data...")
        tracking = load_all_tracking_data(args.tracking_path)
        mappings = load_mappings(args.mapping_path)
        shots = load_all_shots(args.shots_path)
        data = {'tracking': tracking, 'mappings': mappings, 'shots': shots}

        # Step 2: Validate data
        print("\n[2/5] Validating data...")
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
        print("\n[3/5] Preprocessing data...")
        processed_df = preprocess_all_data(
            tracking_data=data['tracking'],
            mappings=data['mappings'],
            shots_data=data['shots']
        )
        features_df = prepare_features_for_inference(processed_df)

        # Step 4: Run inference
        print("\n[4/5] Running inference...")
        results = run_inference_pipeline(
            features_df,
            model_path=args.model_path,
            output_dir=args.output_path
        )

        # Step 5: Output report
        print("\n[5/5] Generating report...")
        report = generate_inference_report(results)
        print(report)

        print("\n==== Pipeline completed successfully! ====")
        print(f"Results saved to: {results['output_file']}")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driblab xG Prediction Pipeline")
    parser.add_argument("--shots_path", required=True, help="Path to shots directory")
    parser.add_argument("--tracking_path", required=True, help="Path to tracking directory")
    parser.add_argument("--mapping_path", required=True, help="Path to mappings directory")
    parser.add_argument("--model_path", required=True, help="Path to model file or directory")
    parser.add_argument("--output_path", required=True, help="Path to output directory")
    args = parser.parse_args()
    main(args)
