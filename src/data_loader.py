import os
import json
import pandas as pd
from typing import List, Dict, Any


def extract_video_timestamps_from_shot_file(filepath: str) -> Dict[str, List[str]]:
    file_key = os.path.basename(filepath).replace(".json", "")
    with open(filepath, 'r') as f:
        data = json.load(f)
    timestamps = [str(event['videoTimestamp']) for event in data if 'videoTimestamp' in event]
    return {file_key: timestamps}


def extract_video_timestamps_from_tracking_file(filepath: str) -> Dict[str, List[str]]:
    file_key = os.path.basename(filepath).replace("_tracking_data.jsonl", "")
    timestamps = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    ts = data.get('videoTimestamp') or data.get('Videotimestamp')
                    if ts is not None:
                        timestamps.append(str(ts))
                except json.JSONDecodeError:
                    continue
    return {file_key: timestamps}


def load_all_tracking_timestamps(tracking_dir: str = "data/shot_pack/jsonls") -> Dict[str, List[str]]:
    all_timestamps = {}
    if not os.path.exists(tracking_dir):
        raise FileNotFoundError(f"Tracking directory not found: {tracking_dir}")
    
    for filename in os.listdir(tracking_dir):
        if filename.endswith('_tracking_data.jsonl'):
            file_path = os.path.join(tracking_dir, filename)
            timestamps_dict = extract_video_timestamps_from_tracking_file(file_path)
            all_timestamps.update(timestamps_dict)
    
    return all_timestamps


def load_all_shots_timestamps(shots_dir: str = "data/shot_pack/shots") -> Dict[str, List[str]]:
    all_timestamps = {}
    if not os.path.exists(shots_dir):
        raise FileNotFoundError(f"Shots directory not found: {shots_dir}")
    
    for filename in os.listdir(shots_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(shots_dir, filename)
            timestamps_dict = extract_video_timestamps_from_shot_file(file_path)
            all_timestamps.update(timestamps_dict)
    
    return all_timestamps


def load_all_tracking_data(tracking_dir: str = "data/shot_pack/jsonls") -> List[Dict[str, Any]]:
    """
    Loads all tracking JSONL files from the given directory.
    
    Args:
        tracking_dir: Path to directory containing tracking JSONL files
        
    Returns:
        List of dictionaries containing tracking data from all files
    """
    tracking_data = []
    
    if not os.path.exists(tracking_dir):
        raise FileNotFoundError(f"Tracking directory not found: {tracking_dir}")
    
    for filename in os.listdir(tracking_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(tracking_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                tracking_data.append(data)
                            except json.JSONDecodeError as e:
                                print(f"Error parsing line {line_num} in {filename}: {e}")
                print(f"Loaded tracking data from: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Total tracking records loaded: {len(tracking_data)}")
    return tracking_data


def load_mappings(mappings_dir: str = "data/shot_pack/mappings") -> Dict[str, Any]:
    """
    Loads both mapping JSON files and returns as a dictionary.
    
    Args:
        mappings_dir: Path to directory containing mapping JSON files
        
    Returns:
        Dictionary containing both mapping files
    """
    mappings = {}
    
    if not os.path.exists(mappings_dir):
        raise FileNotFoundError(f"Mappings directory not found: {mappings_dir}")
    
    mapping_files = [
        "player_event_id_to_tracking_id.json",
        "team_event_id_to_tracking_id.json"
    ]
    
    for filename in mapping_files:
        file_path = os.path.join(mappings_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    key = filename.replace('.json', '')
                    mappings[key] = data
                print(f"Loaded mapping: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Warning: Mapping file not found: {filename}")
    
    return mappings


def load_all_shots(shots_dir: str = "data/shot_pack/shots") -> List[Dict[str, Any]]:
    """
    Loads all shot JSON files from the given directory.
    
    Args:
        shots_dir: Path to directory containing shot JSON files
        
    Returns:
        List of dictionaries containing shot data from all files
    """
    shots_data = []
    
    if not os.path.exists(shots_dir):
        raise FileNotFoundError(f"Shots directory not found: {shots_dir}")
    
    for filename in os.listdir(shots_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(shots_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        shots_data.extend(data)
                    else:
                        shots_data.append(data)
                print(f"Loaded shots data from: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Total shot records loaded: {len(shots_data)}")
    return shots_data


def load_all_data(base_dir: str = "data/shot_pack") -> Dict[str, Any]:
    """
    Loads all data from the shot_pack directory structure.
    
    Args:
        base_dir: Base directory containing jsonls, mappings, and shots folders
        
    Returns:
        Dictionary containing all loaded data
    """
    data = {}
    
    tracking_dir = os.path.join(base_dir, "jsonls")
    data['tracking'] = load_all_tracking_data(tracking_dir)
    
    mappings_dir = os.path.join(base_dir, "mappings")
    data['mappings'] = load_mappings(mappings_dir)
    
    shots_dir = os.path.join(base_dir, "shots")
    data['shots'] = load_all_shots(shots_dir)
    
    return data


if __name__ == "__main__":
    try:
        all_data = load_all_data()
        print("\nData loading summary:")
        print(f"Tracking records: {len(all_data['tracking'])}")
        print(f"Mapping files: {len(all_data['mappings'])}")
        print(f"Shot records: {len(all_data['shots'])}")
    except Exception as e:
        print(f"Error during data loading: {e}")