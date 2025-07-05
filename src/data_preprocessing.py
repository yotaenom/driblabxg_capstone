from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def preprocess_tracking_data(tracking_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Preprocesses tracking data.
    
    Args:
        tracking_data: List of dictionaries containing tracking data
        
    Returns:
        Preprocessed tracking data as DataFrame
    """
    if not tracking_data:
        raise ValueError("Tracking data is empty")
    
    # Convert to DataFrame
    df = pd.DataFrame(tracking_data)
    
    # TODO: Add tracking-specific preprocessing logic
    # Examples:
    # - Handle missing values
    # - Convert data types
    # - Filter invalid coordinates
    # - Normalize coordinates
    # - Add derived features
    
    # Placeholder: Basic missing value handling
    # df = df.dropna(subset=['x', 'y'])  # Remove rows with missing coordinates
    
    print(f"✓ Tracking data preprocessed: {len(df)} records")
    return df


def preprocess_mappings(mappings: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Preprocesses mappings data.
    
    Args:
        mappings: Dictionary containing mapping data
        
    Returns:
        Dictionary of preprocessed mapping DataFrames
    """
    if not mappings:
        raise ValueError("Mappings data is empty")
    
    processed_mappings = {}
    
    for mapping_name, mapping_data in mappings.items():
        # Convert to DataFrame if it's not already
        if isinstance(mapping_data, dict):
            df = pd.DataFrame(list(mapping_data.items()), columns=['key', 'value'])
        elif isinstance(mapping_data, list):
            df = pd.DataFrame(mapping_data)
        else:
            df = pd.DataFrame(mapping_data)
        
        # TODO: Add mapping-specific preprocessing logic
        # Examples:
        # - Handle missing values
        # - Convert data types
        # - Validate mapping relationships
        # - Create lookup dictionaries
        
        processed_mappings[mapping_name] = df
        print(f"✓ Mapping '{mapping_name}' preprocessed: {len(df)} records")
    
    return processed_mappings


def preprocess_shots_data(shots_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Preprocesses shots data using proper one-hot encoding.
    
    Args:
        shots_data: List of dictionaries containing shots data
        
    Returns:
        Preprocessed shots data as DataFrame with final selected features
    """
    import pandas as pd
    from sklearn.preprocessing import MultiLabelBinarizer
    
    if not shots_data:
        raise ValueError("Shots data is empty")
    
    # Flatten JSON data (same as pd.json_normalize in notebook)
    df = pd.json_normalize(shots_data, sep='_')
    
    # Validate required columns exist
    required_source_columns = [
        'id', 'matchId', 'matchTimestamp', 'videoTimestamp', 
        'location_x', 'location_y', 'team_id', 'player_id',
        'shot_bodyPart', 'assist_info_type_secondary', 'possession_types'
    ]
    
    missing_cols = [col for col in required_source_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    
    df_bodypart = pd.get_dummies(df['shot_bodyPart'], prefix='shot_bodyPart')
    df = pd.concat([df, df_bodypart], axis=1)
    
    # Create shot_bodyPart_head_or_other from the one-hot encoded columns
    head_cols = [col for col in df_bodypart.columns if 'head' in col.lower() or 'other' in col.lower()]
    if head_cols:
        df['shot_bodyPart_head_or_other'] = df[head_cols].max(axis=1).astype(int)
    else:
        # Fallback if no head/other columns found
        df['shot_bodyPart_head_or_other'] = 0
    
    # One-hot encode assist_info_type_secondary using MultiLabelBinarizer
    # Handle NaN values in list columns
    df['assist_info_type_secondary'] = df['assist_info_type_secondary'].fillna('').apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    mlb_assist = MultiLabelBinarizer()
    assist_encoded = mlb_assist.fit_transform(df['assist_info_type_secondary'])
    assist_df = pd.DataFrame(assist_encoded, 
                            columns=[f"assist_type_{col}" for col in mlb_assist.classes_],
                            index=df.index)
    df = pd.concat([df, assist_df], axis=1)
    
    # Find and extract assist type columns (search for patterns)
    assist_cols = assist_df.columns.tolist()
    
    # Look for long_pass column
    long_pass_cols = [col for col in assist_cols if 'long_pass' in col.lower()]
    df['assist_type_long_pass'] = (
        assist_df[long_pass_cols[0]].astype(int) if long_pass_cols 
        else pd.Series(0, index=df.index, dtype=int)
    )
    
    # Look for through_pass column  
    through_pass_cols = [col for col in assist_cols if 'through_pass' in col.lower()]
    df['assist_type_through_pass'] = (
        assist_df[through_pass_cols[0]].astype(int) if through_pass_cols
        else pd.Series(0, index=df.index, dtype=int)
    )
    
    # One-hot encode possession_types using MultiLabelBinarizer  
    # Handle NaN values in list columns
    df['possession_types'] = df['possession_types'].fillna('').apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    mlb_possession = MultiLabelBinarizer()
    possession_encoded = mlb_possession.fit_transform(df['possession_types'])
    possession_df = pd.DataFrame(possession_encoded,
                                columns=mlb_possession.classes_,
                                index=df.index)
    df = pd.concat([df, possession_df], axis=1)
    
    # Look for direct_free_kick column
    possession_cols = possession_df.columns.tolist()
    free_kick_cols = [col for col in possession_cols if 'direct_free_kick' in col.lower()]
    df['direct_free_kick'] = (
        possession_df[free_kick_cols[0]].astype(int) if free_kick_cols
        else pd.Series(0, index=df.index, dtype=int)
    )

    
    # 4. Select final features
    final_columns = [
        'id', 'matchId', 'matchTimestamp', 'videoTimestamp',
        'location_x', 'location_y', 'team_id', 'player_id',
        'shot_bodyPart_head_or_other', 'assist_type_long_pass', 
        'direct_free_kick', 'assist_type_through_pass'
    ]
    
    # Create final dataframe with selected columns
    final_df = df[final_columns].copy()
    
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
    
    # 6. Handle missing values
    
    # Drop rows with missing coordinates (critical for analysis)
    initial_rows = len(final_df)
    final_df = final_df.dropna(subset=['location_x', 'location_y'])
    dropped_rows = initial_rows - len(final_df)
    
    
    # Fill missing timestamps with empty string if needed
    timestamp_cols = ['matchTimestamp', 'videoTimestamp']
    for col in timestamp_cols:
        final_df[col] = final_df[col].fillna('')
    
    
    
    return final_df


def merge_data(tracking_df: pd.DataFrame, 
               mappings: Dict[str, pd.DataFrame], 
               shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges all preprocessed data into a single DataFrame for modeling.
    
    Args:
        tracking_df: Preprocessed tracking data
        mappings: Dictionary of preprocessed mapping DataFrames
        shots_df: Preprocessed shots data
        
    Returns:
        Merged DataFrame ready for inference
    """
    print("Starting data merge...")
    
    # TODO: Implement merging logic based on your data structure
    # Examples:
    # - Join shots with tracking data using event IDs
    # - Join with player/team mappings
    # - Create features from tracking data for each shot
    # - Aggregate tracking data per shot
    
    # Placeholder: Simple concatenation (replace with actual merge logic)
    # merged_df = pd.concat([shots_df, tracking_df], axis=1, join='inner')
    
    # For now, return shots data as the base (replace with actual merge)
    merged_df = shots_df.copy()
    
    print(f"✓ Data merged successfully: {len(merged_df)} final records")
    return merged_df


def preprocess_all_data(tracking_data: List[Dict[str, Any]], 
                       mappings: Dict[str, Any], 
                       shots_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for all data types.
    
    Args:
        tracking_data: Raw tracking data
        mappings: Raw mappings data
        shots_data: Raw shots data
        
    Returns:
        Preprocessed and merged DataFrame ready for inference
    """
    print("Starting data preprocessing pipeline...")
    
    # Step 1: Preprocess each data type
    tracking_df = preprocess_tracking_data(tracking_data)
    processed_mappings = preprocess_mappings(mappings)
    shots_df = preprocess_shots_data(shots_data)
    
    # Step 2: Merge all data
    final_df = merge_data(tracking_df, processed_mappings, shots_df)
    
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