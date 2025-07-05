from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def merge_raw_data(tracking_data: List[Dict[str, Any]], 
                   mappings: Dict[str, Any], 
                   shots_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merges raw data from all sources into a single DataFrame before preprocessing.
    
    Args:
        tracking_data: Raw tracking data
        mappings: Raw mappings data
        shots_data: Raw shots data
        
    Returns:
        Merged raw DataFrame
    """
    print("Starting raw data merge...")
    
    # Convert shots data to DataFrame
    if not shots_data:
        raise ValueError("Shots data is empty")
    shots_df = pd.json_normalize(shots_data, sep='_')
    
    # Convert tracking data to DataFrame
    tracking_df = pd.DataFrame()
    if tracking_data:
        tracking_df = pd.DataFrame(tracking_data)
    
    # Process mappings
    processed_mappings = {}
    if mappings:
        for mapping_name, mapping_data in mappings.items():
            if isinstance(mapping_data, dict):
                df = pd.DataFrame(list(mapping_data.items()), columns=['key', 'value'])
            elif isinstance(mapping_data, list):
                df = pd.DataFrame(mapping_data)
            else:
                df = pd.DataFrame(mapping_data)
            processed_mappings[mapping_name] = df
    
    # TODO: Implement actual merging logic based on your data structure
    # For now, we'll use shots data as the base and add tracking/mapping data
    # You should replace this with your specific merge logic
    
    # Placeholder: Use shots data as base (replace with actual merge logic)
    merged_df = shots_df.copy()
    
    # Add tracking data if available (you'll need to implement proper join logic)
    if not tracking_df.empty:
        # TODO: Join tracking data with shots data using appropriate keys
        # Example: merged_df = merged_df.merge(tracking_df, on='event_id', how='left')
        pass
    
    # Add mapping data if available (you'll need to implement proper join logic)
    for mapping_name, mapping_df in processed_mappings.items():
        # TODO: Join mapping data with merged_df using appropriate keys
        # Example: merged_df = merged_df.merge(mapping_df, on='player_id', how='left')
        pass
    
    print(f"✓ Raw data merged: {len(merged_df)} records")
    return merged_df


def preprocess_merged_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the merged dataset with all the feature engineering and cleaning.
    
    Args:
        merged_df: Merged raw DataFrame
        
    Returns:
        Preprocessed DataFrame with final selected features
    """
    print("Starting preprocessing of merged data...")
    
    from sklearn.preprocessing import MultiLabelBinarizer
    
    # Validate required columns exist
    required_source_columns = [
        'id', 'matchId', 'matchTimestamp', 'videoTimestamp', 
        'location_x', 'location_y', 'team_id', 'player_id',
        'shot_bodyPart', 'assist_info_type_secondary', 'possession_types'
    ]
    
    missing_cols = [col for col in required_source_columns if col not in merged_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # One-hot encode shot_bodyPart
    df_bodypart = pd.get_dummies(merged_df['shot_bodyPart'], prefix='shot_bodyPart')
    merged_df = pd.concat([merged_df, df_bodypart], axis=1)
    
    # Create shot_bodyPart_head_or_other from the one-hot encoded columns
    head_cols = [col for col in df_bodypart.columns if 'head' in col.lower() or 'other' in col.lower()]
    if head_cols:
        merged_df['shot_bodyPart_head_or_other'] = merged_df[head_cols].max(axis=1).astype(int)
    else:
        # Fallback if no head/other columns found
        merged_df['shot_bodyPart_head_or_other'] = 0
    
    # One-hot encode assist_info_type_secondary using MultiLabelBinarizer
    # Handle NaN values in list columns
    merged_df['assist_info_type_secondary'] = merged_df['assist_info_type_secondary'].fillna('').apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    mlb_assist = MultiLabelBinarizer()
    assist_encoded = mlb_assist.fit_transform(merged_df['assist_info_type_secondary'])
    assist_df = pd.DataFrame(assist_encoded, 
                            columns=[f"assist_type_{col}" for col in mlb_assist.classes_],
                            index=merged_df.index)
    merged_df = pd.concat([merged_df, assist_df], axis=1)
    
    # Find and extract assist type columns (search for patterns)
    assist_cols = assist_df.columns.tolist()
    
    # Look for long_pass column
    long_pass_cols = [col for col in assist_cols if 'long_pass' in col.lower()]
    merged_df['assist_type_long_pass'] = (
        assist_df[long_pass_cols[0]].astype(int) if long_pass_cols 
        else pd.Series(0, index=merged_df.index, dtype=int)
    )
    
    # Look for through_pass column  
    through_pass_cols = [col for col in assist_cols if 'through_pass' in col.lower()]
    merged_df['assist_type_through_pass'] = (
        assist_df[through_pass_cols[0]].astype(int) if through_pass_cols
        else pd.Series(0, index=merged_df.index, dtype=int)
    )
    
    # One-hot encode possession_types using MultiLabelBinarizer  
    # Handle NaN values in list columns
    merged_df['possession_types'] = merged_df['possession_types'].fillna('').apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    mlb_possession = MultiLabelBinarizer()
    possession_encoded = mlb_possession.fit_transform(merged_df['possession_types'])
    possession_df = pd.DataFrame(possession_encoded,
                                columns=mlb_possession.classes_,
                                index=merged_df.index)
    merged_df = pd.concat([merged_df, possession_df], axis=1)
    
    # Look for direct_free_kick column
    possession_cols = possession_df.columns.tolist()
    free_kick_cols = [col for col in possession_cols if 'direct_free_kick' in col.lower()]
    merged_df['direct_free_kick'] = (
        possession_df[free_kick_cols[0]].astype(int) if free_kick_cols
        else pd.Series(0, index=merged_df.index, dtype=int)
    )

    # Select final features
    final_columns = [
        'id', 'matchId', 'matchTimestamp', 'videoTimestamp',
        'location_x', 'location_y', 'team_id', 'player_id',
        'shot_bodyPart_head_or_other', 'assist_type_long_pass', 
        'direct_free_kick', 'assist_type_through_pass'
    ]
    
    # Create final dataframe with selected columns
    final_df = merged_df[final_columns].copy()
    
    # Data type conversions and cleaning
    
    # Ensure numeric columns are proper type
    numeric_cols = ['location_x', 'location_y']
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    # Ensure ID columns are strings
    id_cols = ['id', 'matchId', 'team_id', 'player_id']
    for col in id_cols:
        final_df[col] = final_df[col].astype(str)
    
    # Ensure binary features are int
    binary_cols = ['shot_bodyPart_head_or_other', 'assist_type_long_pass', 
                   'direct_free_kick', 'assist_type_through_pass']
    for col in binary_cols:
        final_df[col] = final_df[col].astype(int)
    
    # Handle missing values
    
    # Drop rows with missing coordinates (critical for analysis)
    initial_rows = len(final_df)
    final_df = final_df.dropna(subset=['location_x', 'location_y'])
    dropped_rows = initial_rows - len(final_df)
    
    if dropped_rows > 0:
        print(f"⚠ Dropped {dropped_rows} rows with missing coordinates")
    
    # Fill missing timestamps with empty string if needed
    timestamp_cols = ['matchTimestamp', 'videoTimestamp']
    for col in timestamp_cols:
        final_df[col] = final_df[col].fillna('')
    
    print(f"✓ Merged data preprocessed: {len(final_df)} final records")
    return final_df


def preprocess_all_data(tracking_data: List[Dict[str, Any]], 
                       mappings: Dict[str, Any], 
                       shots_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Complete preprocessing pipeline: merge first, then preprocess.
    
    Args:
        tracking_data: Raw tracking data
        mappings: Raw mappings data
        shots_data: Raw shots data
        
    Returns:
        Preprocessed and merged DataFrame ready for inference
    """
    print("Starting data preprocessing pipeline...")
    
    # Step 1: Merge all raw data first
    merged_df = merge_raw_data(tracking_data, mappings, shots_data)
    
    # Step 2: Preprocess the merged dataset
    final_df = preprocess_merged_data(merged_df)
    
    print("✓ Data preprocessing pipeline completed!")
    return final_df


def prepare_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the final DataFrame for model inference.
    
    Args:
        df: Merged and preprocessed DataFrame
        
    Returns:
        DataFrame with features ready for model prediction
    """
    # TODO: Add final feature preparation logic
    # Examples:
    # - Select only feature columns
    # - Handle categorical variables (encoding)
    # - Scale numerical features
    # - Create final feature matrix
    
    # Placeholder: Return as-is (replace with actual feature preparation)
    features_df = df.copy()
    
    print(f"✓ Features prepared for inference: {features_df.shape}")
    return features_df


# Keep the old functions for backward compatibility but mark them as deprecated
def preprocess_tracking_data(tracking_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    DEPRECATED: Use merge_raw_data and preprocess_merged_data instead.
    """
    print("WARNING: preprocess_tracking_data is deprecated. Use merge_raw_data and preprocess_merged_data instead.")
    return pd.DataFrame()


def preprocess_mappings(mappings: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    DEPRECATED: Use merge_raw_data and preprocess_merged_data instead.
    """
    print("WARNING: preprocess_mappings is deprecated. Use merge_raw_data and preprocess_merged_data instead.")
    return {}


def preprocess_shots_data(shots_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    DEPRECATED: Use merge_raw_data and preprocess_merged_data instead.
    """
    print("WARNING: preprocess_shots_data is deprecated. Use merge_raw_data and preprocess_merged_data instead.")
    return pd.DataFrame()


def merge_data(tracking_df: pd.DataFrame, 
               mappings: Dict[str, pd.DataFrame], 
               shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: Use merge_raw_data and preprocess_merged_data instead.
    """
    print("WARNING: merge_data is deprecated. Use merge_raw_data and preprocess_merged_data instead.")
    return pd.DataFrame()


if __name__ == "__main__":
    # Test the preprocessing pipeline
    from src.data_loader import load_all_data
    
    try:
        # Load data
        data = load_all_data()
        
        # Run preprocessing pipeline
        processed_df = preprocess_all_data(
            tracking_data=data['tracking'],
            mappings=data['mappings'],
            shots_data=data['shots']
        )
        
        # Prepare features for inference
        features_df = prepare_features_for_inference(processed_df)
        
        print(f"\nPreprocessing Summary:")
        print(f"Final dataset shape: {features_df.shape}")
        print(f"Columns: {list(features_df.columns)}")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}") 