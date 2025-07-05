from typing import List, Dict, Any, Set
import pandas as pd


def validate_tracking_schema(tracking_data: List[Dict[str, Any]]) -> bool:
    """
    Validates the schema for tracking data.
    
    Args:
        tracking_data: List of dictionaries containing tracking data
        
    Returns:
        True if validation passes, raises AssertionError otherwise
    """
    if not tracking_data:
        raise AssertionError("Tracking data is empty")
    
    # Get all unique keys from the tracking data
    all_keys = set()
    for record in tracking_data:
        if isinstance(record, dict):
            all_keys.update(record.keys())
    
    # TODO: Define expected columns for tracking data
    # Replace this with actual expected columns
    expected_columns: Set[str] = set()  # e.g., {'frame', 'player_id', 'x', 'y', 'team_id'}
    
    # Check if all expected columns are present
    missing_columns = expected_columns - all_keys
    if missing_columns:
        raise AssertionError(f"Tracking data missing required columns: {missing_columns}")
    
    print("✓ Tracking data schema validation passed")
    return True


def validate_mappings_schema(mappings: Dict[str, Any]) -> bool:
    """
    Validates the schema for mappings data.
    
    Args:
        mappings: Dictionary containing mapping data
        
    Returns:
        True if validation passes, raises AssertionError otherwise
    """
    if not mappings:
        raise AssertionError("Mappings data is empty")
    
    # Validate each mapping file
    for mapping_name, mapping_data in mappings.items():
        if not mapping_data:
            raise AssertionError(f"Mapping '{mapping_name}' is empty")
        
        # TODO: Define expected structure for each mapping type
        # Replace these with actual expected structures
        if mapping_name == "player_event_id_to_tracking_id":
            # Example: expected_structure = {'player_id': 'tracking_id'}
            pass
        elif mapping_name == "team_event_id_to_tracking_id":
            # Example: expected_structure = {'team_id': 'tracking_id'}
            pass
    
    print("✓ Mappings schema validation passed")
    return True


def validate_shots_schema(shots_data: List[Dict[str, Any]]) -> bool:
    """
    Validates the schema for shots data.
    
    Args:
        shots_data: List of dictionaries containing shots data
        
    Returns:
        True if validation passes, raises AssertionError otherwise
    """
    if not shots_data:
        raise AssertionError("Shots data is empty")
    
    # Get all unique keys from the shots data
    all_keys = set()
    for record in shots_data:
        if isinstance(record, dict):
            all_keys.update(record.keys())
    
    # Expected base columns for final feature extraction
    # Note: Raw JSON data will have many more columns - this is expected
    expected_columns: Set[str] = {
        # Basic identifiers and metadata
        'id',
        'matchId',
        'matchTimestamp',
        'videoTimestamp',
        
        # Location data (required for shot analysis)
        'location_x',
        'location_y',
        
        # Team and player identifiers
        'team_id',
        'player_id',
        
        # Source columns for feature engineering
        'shot_bodyPart',           # -> shot_bodyPart_head_or_other
        'assist_info_type_secondary',  # -> assist_type_long_pass, assist_type_through_pass
        'possession_types'         # -> direct_free_kick
    }
    
    # Check if all expected columns are present (extra columns are fine)
    missing_columns = expected_columns - all_keys
    if missing_columns:
        raise AssertionError(f"Shots data missing required columns: {missing_columns}")
    
    # Report on what we found
    extra_columns = all_keys - expected_columns
    
    # Validate data types for critical columns
    sample_record = shots_data[0]
    
    # Check numeric fields
    numeric_fields = ['location_x', 'location_y']
    for field in numeric_fields:
        if field in sample_record and sample_record[field] is not None:
            if not isinstance(sample_record[field], (int, float)):
                raise AssertionError(f"Field '{field}' should be numeric, got {type(sample_record[field])}")
    
    # Check string/ID fields
    id_fields = ['id', 'matchId', 'team_id', 'player_id']
    for field in id_fields:
        if field in sample_record and sample_record[field] is not None:
            if not isinstance(sample_record[field], (str, int)):
                raise AssertionError(f"Field '{field}' should be string or int, got {type(sample_record[field])}")
    
    # Check string fields
    if 'shot_bodyPart' in sample_record and sample_record['shot_bodyPart'] is not None:
        if not isinstance(sample_record['shot_bodyPart'], str):
            raise AssertionError(f"Field 'shot_bodyPart' should be string, got {type(sample_record['shot_bodyPart'])}")
    
    # Check list fields
    list_fields = ['assist_info_type_secondary', 'possession_types']
    for field in list_fields:
        if field in sample_record and sample_record[field] is not None:
            if not isinstance(sample_record[field], list):
                raise AssertionError(f"Field '{field}' should be list, got {type(sample_record[field])}")
    
    return True


def validate_all_data(tracking_data: List[Dict[str, Any]], 
                     mappings: Dict[str, Any], 
                     shots_data: List[Dict[str, Any]]) -> bool:
    """
    Validates schema for all three data types.
    
    Args:
        tracking_data: List of dictionaries containing tracking data
        mappings: Dictionary containing mapping data
        shots_data: List of dictionaries containing shots data
        
    Returns:
        True if all validations pass, raises AssertionError otherwise
    """
    print("Starting data validation...")
    
    # Validate each data type
    validate_tracking_schema(tracking_data)
    validate_mappings_schema(mappings)
    validate_shots_schema(shots_data)
    
    print("✓ All data validation passed!")
    return True


def validate_data_types(tracking_data: List[Dict[str, Any]], 
                       mappings: Dict[str, Any], 
                       shots_data: List[Dict[str, Any]]) -> bool:
    """
    Validates data types for all three data types.
    This is a placeholder for future type validation logic.
    
    Args:
        tracking_data: List of dictionaries containing tracking data
        mappings: Dictionary containing mapping data
        shots_data: List of dictionaries containing shots data
        
    Returns:
        True if all type validations pass, raises AssertionError otherwise
    """
    # TODO: Add specific data type validation logic
    # Example: Check that 'x' and 'y' are numeric, 'player_id' is string, etc.
    
    print("✓ Data type validation passed (placeholder)")
    return True


if __name__ == "__main__":
    # Test the validation framework
    from src.data_loader import load_all_data
    
    try:
        # Load data
        data = load_all_data()
        
        # Validate schema
        validate_all_data(
            tracking_data=data['tracking'],
            mappings=data['mappings'],
            shots_data=data['shots']
        )
        
        # Validate data types
        validate_data_types(
            tracking_data=data['tracking'],
            mappings=data['mappings'],
            shots_data=data['shots']
        )
        
    except Exception as e:
        print(f"Validation failed: {e}")
