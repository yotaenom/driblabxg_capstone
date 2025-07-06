from typing import List, Dict, Any, Set
import pandas as pd
try:
    from typing import Union
except ImportError:
    from typing_extensions import Union


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


def validate_shots_schema(shots_data: Union[List[Dict[str, Any]], pd.DataFrame]) -> bool:
    """
    Validates the schema for shots data.
    
    Args:
        shots_data: List of dictionaries containing shots data OR pandas DataFrame
        
    Returns:
        True if validation passes, raises AssertionError otherwise
    """
    if shots_data is None or (hasattr(shots_data, '__len__') and len(shots_data) == 0):
        raise AssertionError("Shots data is empty")
    
    # Handle both DataFrame and list of dicts
    if isinstance(shots_data, pd.DataFrame):
        # For DataFrames, check for flattened column names
        all_keys = set(shots_data.columns)
        sample_record = shots_data.iloc[0].to_dict() if len(shots_data) > 0 else {}
        is_dataframe = True
        is_flattened = True
    else:
        # For list of dicts, check if it's raw JSON (nested) or already flattened
        all_keys = set()
        for record in shots_data:
            if isinstance(record, dict):
                all_keys.update(record.keys())
        sample_record = shots_data[0] if shots_data else {}
        is_dataframe = False
        
        # Detect if data is already flattened (has underscore-separated names)
        flattened_indicators = ['location_x', 'team_id', 'player_id', 'shot_bodyPart']
        is_flattened = any(key in all_keys for key in flattened_indicators)
    
    if is_flattened:
        # Check for flattened column names (after pd.json_normalize)
        expected_columns: Set[str] = {
            # Basic identifiers and metadata
            'id', 'matchId', 'matchTimestamp', 'videoTimestamp',
            
            # Flattened location data
            'location_x', 'location_y',
            
            # Flattened team and player identifiers  
            'team_id', 'player_id',
            
            # Flattened source columns for feature engineering
            'shot_bodyPart',                    # -> shot_bodyPart_head_or_other
            'assist_info_type_secondary',       # -> assist_type_long_pass, assist_type_through_pass
            'possession_types'                  # -> direct_free_kick
        }
        
        # Validate data types for flattened data
        def validate_flattened_types():
            # Check numeric fields
            numeric_fields = ['location_x', 'location_y']
            for field in numeric_fields:
                if field in sample_record and pd.notna(sample_record[field]):
                    value = sample_record[field]
                    if not (isinstance(value, (int, float)) or pd.api.types.is_numeric_dtype(type(value))):
                        raise AssertionError(f"Field '{field}' should be numeric, got {type(value)}")
            
            # Check string/ID fields
            id_fields = ['id', 'matchId', 'team_id', 'player_id']
            for field in id_fields:
                if field in sample_record and pd.notna(sample_record[field]):
                    value = sample_record[field]
                    if not (isinstance(value, (str, int)) or pd.api.types.is_string_dtype(type(value)) or pd.api.types.is_integer_dtype(type(value))):
                        raise AssertionError(f"Field '{field}' should be string or int, got {type(value)}")
            
            # Check shot_bodyPart 
            if 'shot_bodyPart' in sample_record and pd.notna(sample_record['shot_bodyPart']):
                value = sample_record['shot_bodyPart']
                if not (isinstance(value, str) or pd.api.types.is_string_dtype(type(value))):
                    raise AssertionError(f"Field 'shot_bodyPart' should be string, got {type(value)}")
            
            # Check list fields
            list_fields = ['assist_info_type_secondary', 'possession_types']
            for field in list_fields:
                if field in sample_record and pd.notna(sample_record[field]):
                    value = sample_record[field]
                    if not isinstance(value, list):
                        raise AssertionError(f"Field '{field}' should be list, got {type(value)}")
        
        validate_flattened_types()
        
    else:
        # Check for raw nested JSON structure (before pd.json_normalize)
        required_columns: Set[str] = {
            # Basic identifiers and metadata (top-level)
            'id', 'matchId', 'matchTimestamp', 'videoTimestamp',
            
            # Nested objects that will be flattened
            'location',        # contains x, y
            'team',           # contains id
            'player',         # contains id  
            'shot',           # contains bodyPart
            'possession'      # contains types
        }
        
        optional_columns: Set[str] = {
            'assist_info',    # contains type_secondary (optional)
        }
        
        expected_columns = required_columns | optional_columns
        
        # Validate nested structure
        def validate_nested_structure():
            if 'location' in sample_record:
                location = sample_record['location']
                if isinstance(location, dict):
                    if 'x' not in location or 'y' not in location:
                        raise AssertionError("location object missing 'x' or 'y' fields")
                    if not isinstance(location.get('x'), (int, float)) or not isinstance(location.get('y'), (int, float)):
                        raise AssertionError("location.x and location.y should be numeric")
            
            if 'team' in sample_record:
                team = sample_record['team']
                if isinstance(team, dict) and 'id' not in team:
                    raise AssertionError("team object missing 'id' field")
            
            if 'player' in sample_record:
                player = sample_record['player']
                if isinstance(player, dict) and 'id' not in player:
                    raise AssertionError("player object missing 'id' field")
            
            if 'shot' in sample_record:
                shot = sample_record['shot']
                if isinstance(shot, dict) and 'bodyPart' not in shot:
                    raise AssertionError("shot object missing 'bodyPart' field")
        
        validate_nested_structure()
    
    # Check if all required columns are present (extra columns are fine)
    missing_required_columns = required_columns - all_keys
    if missing_required_columns:
        raise AssertionError(f"Shots data missing required columns: {missing_required_columns}")
    
    # Check for optional columns and warn if missing
    missing_optional_columns = optional_columns - all_keys
    if missing_optional_columns:
        print(f"⚠ Warning: Missing optional columns: {missing_optional_columns}")
    
    # Report on what we found
    extra_columns = all_keys - expected_columns
    
    
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
