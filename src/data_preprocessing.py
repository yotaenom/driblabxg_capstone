import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
#from scipy.spatial import Voronoi
#from shapely.geometry import Polygon


def get_closest_tracking_frames_for_shots_with_context(shot_file, tracking_file, prev_frames=10, next_frames=10):
    """
    Extract closest tracking frames for shot timestamps, including previous and next frames around each match.
    """
    with open(shot_file, 'r') as sf:
        shot_data = json.load(sf)
    
    video_timestamps = [
        str(event['videoTimestamp']) for event in shot_data 
        if 'videoTimestamp' in event and event.get('type', {}).get('primary') == 'shot'
    ]
    
    tracking_frames = []
    with open(tracking_file, 'r') as tf:
        for line in tf:
            try:
                frame = json.loads(line)
                ts = frame.get('videoTimestamp') or frame.get('Videotimestamp')
                if ts is not None:
                    tracking_frames.append(frame)
            except json.JSONDecodeError:
                continue
    
    tracking_frames.sort(key=lambda x: x.get('videoTimestamp') or x.get('Videotimestamp'))
    
    closest_frames_with_context = {}
    
    for shot_ts in video_timestamps:
        shot_timestamp = float(shot_ts)
        min_diff = float('inf')
        closest_index = -1
        
        for i, frame in enumerate(tracking_frames):
            frame_ts = frame.get('videoTimestamp') or frame.get('Videotimestamp')
            diff = abs(frame_ts - shot_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_index = i
        
        if closest_index != -1:
            prev_start_idx = max(0, closest_index - prev_frames)
            prev_frames_data = []
            for i in range(prev_start_idx, closest_index):
                frame_info = {
                    'frame': tracking_frames[i],
                    'position_relative_to_closest': i - closest_index,
                    'frame_type': 'previous'
                }
                prev_frames_data.append(frame_info)
            
            next_end_idx = min(len(tracking_frames), closest_index + next_frames + 1)
            next_frames_data = []
            for i in range(closest_index + 1, next_end_idx):
                frame_info = {
                    'frame': tracking_frames[i],
                    'position_relative_to_closest': i - closest_index,
                    'frame_type': 'next'
                }
                next_frames_data.append(frame_info)
            
            closest_frames_with_context[shot_ts] = {
                'closest_frame': tracking_frames[closest_index],
                'previous_frames': prev_frames_data,
                'next_frames': next_frames_data
            }
    
    return closest_frames_with_context


def get_all_shot_tracking_matches_with_context(shots_dir, tracking_dir, prev_frames=10, next_frames=10, max_files=None):
    """
    Process shot-tracking matches with context frames for all files.
    """
    all_matches = {}
    processed_count = 0
    
    json_files = [f for f in os.listdir(shots_dir) if f.endswith('.json')]
    json_files.sort()
    
    if max_files is None:
        max_files = len(json_files)
    
    for filename in json_files:
        if processed_count >= max_files:
            break
            
        match_id = filename.replace('.json', '')
        shot_path = os.path.join(shots_dir, f"{match_id}.json")
        tracking_path = os.path.join(tracking_dir, f"{match_id}_tracking_data.jsonl")
        
        if os.path.exists(tracking_path):
            result = get_closest_tracking_frames_for_shots_with_context(
                shot_path, tracking_path, prev_frames, next_frames
            )
            all_matches[match_id] = result
            processed_count += 1
    
    return all_matches


def extract_match_data_from_directory(directory_path):
    """
    Create a dictionary where keys are match_ids and values are match_data
    from all JSONL files in the specified directory.
    """
    match_data_dict = {}
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    
    for filename in os.listdir(directory_path):
        if not filename.endswith('_tracking_data.jsonl'):
            continue
            
        file_path = os.path.join(directory_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().strip()
                
                if first_line:
                    data = json.loads(first_line)
                    
                    if 'match_data' in data:
                        match_id = data['match_data'].get('match_id')
                        
                        if match_id:
                            match_id_str = str(match_id)
                            match_data_dict[match_id_str] = data
                        
        except (json.JSONDecodeError, Exception):
            continue
    
    return match_data_dict


def compute_shot_to_goal_distance(row, event_to_tracking_map):
    frame = row['closest_frame']
    players_data = row['match_info'].get('players_data', {})
    frame_data = frame['data']
    
    if len(frame_data) != 2:
        return np.nan

    team0_id, team1_id = frame_data.keys()
    team0 = frame_data[team0_id]
    team1 = frame_data[team1_id]

    shooter_event_id = str(row['player_id'])
    tracking_id = str(event_to_tracking_map.get(shooter_event_id))
    shooter_coords = None

    if tracking_id and tracking_id != 'None':
        for team in [team0, team1]:
            for player in team:
                if str(player['id']) == tracking_id:
                    shooter_coords = (player['x'], player['y'])
                    break
            if shooter_coords:
                break

    if not shooter_coords:
        return np.nan

    shooter_team_id = str(row['team_id'])
    shooter_team = team0 if team0_id == shooter_team_id else team1

    gk_coords = None
    for p in shooter_team:
        pid = str(p['id'])
        if pid in players_data.get(shooter_team_id, {}) and players_data[shooter_team_id][pid].get('position') == 'GK':
            gk_coords = (p['x'], p['y'])
            break

    if gk_coords:
        attack_right = shooter_coords[0] > gk_coords[0]
    else:
        attack_right = shooter_coords[0] > 52.5

    goal_x = 105 if attack_right else 0
    goal_y = 34

    dx = shooter_coords[0] - goal_x
    dy = shooter_coords[1] - goal_y
    return np.sqrt(dx**2 + dy**2)


def get_goal_facing_cone(row, event_to_tracking_map):
    frame = row['closest_frame']
    players_data = row['match_info'].get('players_data', {})
    frame_data = frame['data']

    if len(frame_data) != 2:
        return None

    team0_id, team1_id = frame_data.keys()
    team0 = frame_data[team0_id]
    team1 = frame_data[team1_id]

    shooter_event_id = str(row['player_id'])
    tracking_id = str(event_to_tracking_map.get(shooter_event_id))
    shooter_coords = None

    if tracking_id and tracking_id != 'None':
        for team in [team0, team1]:
            for player in team:
                if str(player['id']) == tracking_id:
                    shooter_coords = np.array([player['x'], player['y']])
                    break
            if shooter_coords is not None:
                break

    if shooter_coords is None:
        return None

    shooter_team_id = str(row['team_id'])
    shooter_team = team0 if team0_id == shooter_team_id else team1

    gk_coords = None
    for p in shooter_team:
        pid = str(p['id'])
        if pid in players_data.get(shooter_team_id, {}) and players_data[shooter_team_id][pid].get('position') == 'GK':
            gk_coords = np.array([p['x'], p['y']])
            break

    attack_right = shooter_coords[0] > gk_coords[0] if gk_coords is not None else shooter_coords[0] > 52.5

    goal_x = 105 if attack_right else 0
    goal_y = 34
    left_post = np.array([goal_x, goal_y - 3.66])
    right_post = np.array([goal_x, goal_y + 3.66])

    vec_left = left_post - shooter_coords
    vec_right = right_post - shooter_coords

    return {
        "shooter": shooter_coords,
        "vec_left": vec_left,
        "vec_right": vec_right,
        "goal_x": goal_x,
        "goal_y": goal_y
    }


def compute_shooting_angle_from_cone(row, event_to_tracking_map):
    cone = get_goal_facing_cone(row, event_to_tracking_map)
    if cone is None:
        return np.nan

    vec_left = cone["vec_left"]
    vec_right = cone["vec_right"]

    norm_left = vec_left / np.linalg.norm(vec_left)
    norm_right = vec_right / np.linalg.norm(vec_right)

    dot = np.dot(norm_left, norm_right)
    dot = np.clip(dot, -1.0, 1.0)

    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def compute_goal_angle(row, event_to_tracking_map):
    """
    Compute the angle between the shot direction and the reference direction to goal.
    This is the 'theta' calculation from your reference code.
    """
    frame = row['closest_frame']
    players_data = row['match_info'].get('players_data', {})
    frame_data = frame['data']

    if len(frame_data) != 2:
        return np.nan

    team0_id, team1_id = frame_data.keys()
    team0 = frame_data[team0_id]
    team1 = frame_data[team1_id]

    shooter_event_id = str(row['player_id'])
    tracking_id = str(event_to_tracking_map.get(shooter_event_id))
    shooter_coords = None

    if tracking_id and tracking_id != 'None':
        for team in [team0, team1]:
            for player in team:
                if str(player['id']) == tracking_id:
                    shooter_coords = np.array([player['x'], player['y']])
                    break
            if shooter_coords is not None:
                break

    if shooter_coords is None:
        return np.nan

    shooter_team_id = str(row['team_id'])
    shooter_team = team0 if team0_id == shooter_team_id else team1

    # Find GK
    gk_coords = None
    for p in shooter_team:
        pid = str(p['id'])
        if pid in players_data.get(shooter_team_id, {}) and players_data[shooter_team_id][pid].get('position') == 'GK':
            gk_coords = np.array([p['x'], p['y']])
            break

    if gk_coords is not None:
        attack_right = shooter_coords[0] > gk_coords[0]
    else:
        attack_right = shooter_coords[0] > 52.5

    goal_x = 105 if attack_right else 0
    goal_y = 34
    goal_center = np.array([goal_x, goal_y])

    shot_vec = goal_center - shooter_coords
    ref_vec = np.array([1, 0]) if attack_right else np.array([-1, 0])

    dot = np.dot(shot_vec, ref_vec)
    norm = np.linalg.norm(shot_vec) * np.linalg.norm(ref_vec)
    angle_rad = np.arccos(dot / norm) if norm > 0 else 0.0
    return np.degrees(angle_rad)


def compute_ddi(row, event_to_tracking_map):
    """
    Compute PDI using the simpler cross product method from your reference code.
    """
    cone = get_goal_facing_cone(row, event_to_tracking_map)
    if cone is None:
        return np.nan

    shooter = cone['shooter']
    vec_left = np.append(cone['vec_left'], 0)   # convert to 3D
    vec_right = np.append(cone['vec_right'], 0)

    frame = row['closest_frame']
    match_info = row['match_info']
    players_data = match_info.get('players_data', {})
    frame_data = frame['data']

    if len(frame_data) != 2:
        return np.nan

    team0_id, team1_id = frame_data.keys()
    team0 = frame_data[team0_id]
    team1 = frame_data[team1_id]

    all_players = team0 + team1
    shooter_tracking_id = str(event_to_tracking_map.get(str(row['player_id'])))
    count = 0

    for player in all_players:
        pid = str(player['id'])

        if pid == shooter_tracking_id:
            continue

        # Determine team and role
        team_id = team0_id if player in team0 else team1_id
        if pid not in players_data.get(team_id, {}):
            continue
        if players_data[team_id][pid].get('position') == 'GK':
            continue

        pos = np.array([player['x'], player['y']])
        vec_to_player = np.append(pos - shooter, 0)

        distance = np.linalg.norm(vec_to_player[:2])
        if distance > 5:
            continue

        # Cross product test
        cross1 = np.cross(vec_left, vec_to_player)[2]
        cross2 = np.cross(vec_to_player, vec_right)[2]

        if cross1 >= 0 and cross2 >= 0:
            count += 1

    return count


def count_players_in_cone(row, event_to_tracking_map):
    """
    Count all players (excluding shooter and GK) whose area intersects the cone 
    from shooter to goal posts.
    """
    cone = get_goal_facing_cone(row, event_to_tracking_map)
    if cone is None:
        return np.nan

    shooter_coords = cone['shooter']
    vec_left = cone['vec_left']
    vec_right = cone['vec_right']

    frame = row['closest_frame']
    match_info = row['match_info']
    players_data = match_info.get('players_data', {})
    frame_data = frame['data']

    if len(frame_data) != 2:
        return np.nan

    team0_id, team1_id = frame_data.keys()
    team0 = frame_data[team0_id]
    team1 = frame_data[team1_id]
    all_players = team0 + team1

    shooter_tracking_id = str(event_to_tracking_map.get(str(row['player_id'])))
    count = 0
    
    vec_left_norm = vec_left / np.linalg.norm(vec_left)
    vec_right_norm = vec_right / np.linalg.norm(vec_right)
    angle_cone = np.arccos(np.clip(np.dot(vec_left_norm, vec_right_norm), -1.0, 1.0))

    for player in all_players:
        pid = str(player['id'])
        
        if pid == shooter_tracking_id:
            continue

        team_id = team0_id if player in team0 else team1_id
        if pid not in players_data.get(team_id, {}):
            continue
        if players_data[team_id][pid].get('position') == 'GK':
            continue

        player_pos = np.array([player['x'], player['y']])
        
        r = 0.5
        angles = np.linspace(0, 2 * np.pi, num=8, endpoint=False)
        circle_points = player_pos + r * np.c_[np.cos(angles), np.sin(angles)]

        for point in circle_points:
            vec_to_point = point - shooter_coords
            vec_point_norm = vec_to_point / np.linalg.norm(vec_to_point)
            angle_left = np.arccos(np.clip(np.dot(vec_left_norm, vec_point_norm), -1.0, 1.0))
            angle_right = np.arccos(np.clip(np.dot(vec_right_norm, vec_point_norm), -1.0, 1.0))
            if angle_left + angle_right <= angle_cone + 1e-6:
                count += 1
                break

    return count


def is_gk_in_target_area(row, event_to_tracking_map):
    """
    Check if the goalkeeper is positioned in the target goal area.
    """
    frame = row['closest_frame']
    match_info = row['match_info']
    players_data = match_info.get('players_data', {})
    frame_data = frame['data']
    
    if len(frame_data) != 2:
        return np.nan
    
    team0_id, team1_id = frame_data.keys()
    team0 = frame_data[team0_id]
    team1 = frame_data[team1_id]
    
    shooter_event_id = str(row['player_id'])
    tracking_id = str(event_to_tracking_map.get(shooter_event_id))
    shooter_coords = None
    gk_coords = None
    
    all_players = team0 + team1
    
    for player in all_players:
        pid = str(player['id'])
        if pid == tracking_id:
            shooter_coords = np.array([player['x'], player['y']])
            break
    
    if shooter_coords is None:
        return np.nan
    
    attack_right = shooter_coords[0] > 52.5
    target_goal_x = 105 if attack_right else 0
    
    closest_gk_distance = float('inf')
    
    for player in all_players:
        pid = str(player['id'])
        
        is_gk = False
        for team_id in [team0_id, team1_id]:
            if pid in players_data.get(team_id, {}) and players_data[team_id][pid].get('position') == 'GK':
                is_gk = True
                break
        
        if is_gk:
            potential_gk_coords = np.array([player['x'], player['y']])
            gk_distance_to_target = abs(potential_gk_coords[0] - target_goal_x)
            
            if gk_distance_to_target < closest_gk_distance:
                closest_gk_distance = gk_distance_to_target
                gk_coords = potential_gk_coords
    
    if shooter_coords is None or gk_coords is None:
        return np.nan
    
    penalty_area_depth = 17.9
    penalty_area_y_min = 14.3
    penalty_area_y_max = 53.7
    
    if target_goal_x == 105:
        area_x_min = 105 - penalty_area_depth
        area_x_max = 105
    else:
        area_x_min = 0
        area_x_max = penalty_area_depth
    
    area_y_min = penalty_area_y_min
    area_y_max = penalty_area_y_max
    
    gk_x, gk_y = gk_coords
    is_in_target_area = (area_x_min <= gk_x <= area_x_max and
                        area_y_min <= gk_y <= area_y_max)
    
    return is_in_target_area


def get_frame_with_valid_gk(row, event_to_tracking_map):
    """
    Return the best available frame where the goalkeeper is valid.
    """
    if bool(row.get("gk_is_valid")) and isinstance(row.get("closest_frame"), dict):
        return row["closest_frame"]

    search_frames = []

    if isinstance(row.get("previous_frames"), list):
        for frame in sorted(row["previous_frames"], key=lambda x: abs(x['position_relative_to_closest'])):
            search_frames.append(frame['frame'])

    if isinstance(row.get("next_frames"), list):
        for frame in sorted(row["next_frames"], key=lambda x: abs(x['position_relative_to_closest'])):
            search_frames.append(frame['frame'])

    for frame in search_frames:
        row_copy = row.copy()
        row_copy["closest_frame"] = frame
        if is_gk_in_target_area(row_copy, event_to_tracking_map):
            return frame

    return np.nan


def compute_gk_line_offset(row, event_to_tracking_map):
    frame = row.get('frame_gk')
    if frame is np.nan or not isinstance(frame, dict):
        return np.nan

    players_data = row['match_info']['players_data']
    frame_data = frame['data']

    shooter_event_id = str(row['player_id'])
    tracking_id = str(event_to_tracking_map.get(shooter_event_id))

    shooter_coords = None
    for team_players in frame_data.values():
        for player in team_players:
            if str(player['id']) == tracking_id:
                shooter_coords = np.array([player['x'], player['y']])
                break
        if shooter_coords is not None:
            break

    if shooter_coords is None:
        return np.nan

    attack_right = shooter_coords[0] > 52.5
    goal_coords = np.array([105, 34]) if attack_right else np.array([0, 34])

    gk_coords = None
    closest_dist = float('inf')
    for team_id, team_players in frame_data.items():
        team_id_str = str(team_id)
        for player in team_players:
            pid = str(player['id'])
            if pid in players_data.get(team_id_str, {}) and players_data[team_id_str][pid].get('position') == 'GK':
                candidate_coords = np.array([player['x'], player['y']])
                dist_to_goal = abs(candidate_coords[0] - goal_coords[0])
                if dist_to_goal < closest_dist:
                    closest_dist = dist_to_goal
                    gk_coords = candidate_coords

    if gk_coords is None:
        return np.nan

    v = goal_coords - shooter_coords
    w = gk_coords - shooter_coords
    cross = abs(np.cross(v, w))
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.nan

    return cross / norm_v


"""def compute_shooter_voronoi_area(row, event_to_tracking_map, pitch_length=105, pitch_width=68):
    try:
        frame_data = row['closest_frame']['data']
        shooter_event_id = str(row['player_id'])
        tracking_id = str(event_to_tracking_map.get(shooter_event_id))

        if tracking_id is None or tracking_id == 'None':
            return np.nan

        positions = []
        ids = []
        for team_players in frame_data.values():
            for player in team_players:
                positions.append([player['x'], player['y']])
                ids.append(str(player['id']))

        positions = np.array(positions)

        if tracking_id not in ids:
            return np.nan

        shooter_index = ids.index(tracking_id)

        vor = Voronoi(positions)

        pitch_polygon = Polygon([
            (0, 0), (pitch_length, 0),
            (pitch_length, pitch_width), (0, pitch_width)
        ])

        region_index = vor.point_region[shooter_index]
        region = vor.regions[region_index]

        if -1 in region or len(region) == 0:
            return 0.0

        poly_points = [vor.vertices[i] for i in region]
        voronoi_poly = Polygon(poly_points)
        clipped_poly = voronoi_poly.intersection(pitch_polygon)

        return clipped_poly.area

    except Exception:
        return np.nan """


def merge_raw_data(shots_dir: str, tracking_dir: str, mappings_dir: str, 
                   prev_frames: int = 10, next_frames: int = 10) -> pd.DataFrame:
    """
    Merges shots data with tracking data using the frame context approach.
    """
    print("Starting raw data merge...")
    
    # Load shots data
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
    
    shots_df = pd.json_normalize(shots_data, sep='_')
    print(f"Loaded {len(shots_df)} shots")
    
    # Get shot-tracking matches with context
    all_tracking_data = get_all_shot_tracking_matches_with_context(
        shots_dir, tracking_dir, prev_frames, next_frames
    )
    
    # Extract match metadata
    match_info = extract_match_data_from_directory(tracking_dir)
    
    # Augment tracking data with match info
    all_data_augmented = {}
    for match_id, timestamp_dict in all_tracking_data.items():
        all_data_augmented[match_id] = all_tracking_data[match_id].copy()
        
        for timestamp, frame_group in timestamp_dict.items():
            if match_id in match_info:
                all_data_augmented[match_id][timestamp]["match_info"] = match_info[match_id]
    
    # Convert videoTimestamp to string for consistent matching
    shots_df['videoTimestamp'] = shots_df['videoTimestamp'].astype(str)
    
    # Load player mapping
    player_mapping_path = os.path.join(mappings_dir, "player_event_id_to_tracking_id.json")
    with open(player_mapping_path, 'r') as f:
        event_to_tracking_map = json.load(f)
    
    # Build lookup: videoTimestamp → full frame info
    timestamp_lookup = {}
    for match in all_data_augmented.values():
        for timestamp, frame_info in match.items():
            timestamp_lookup[timestamp] = frame_info
    
    # Extract frame parts
    def extract_frame_part(row, part):
        frame_data = timestamp_lookup.get(row['videoTimestamp'], {})
        return frame_data.get(part)
    
    # Assign all frame-related columns
    shots_df['closest_frame'] = shots_df.apply(lambda row: extract_frame_part(row, 'closest_frame'), axis=1)
    shots_df['previous_frames'] = shots_df.apply(lambda row: extract_frame_part(row, 'previous_frames'), axis=1)
    shots_df['next_frames'] = shots_df.apply(lambda row: extract_frame_part(row, 'next_frames'), axis=1)
    shots_df['match_info'] = shots_df['videoTimestamp'].map(lambda ts: timestamp_lookup.get(ts, {}).get('match_info'))
    
    # Filter out rows without tracking data
    shots_with_tracking = shots_df.dropna(subset=['closest_frame'])
    print(f"Shots with tracking data: {len(shots_with_tracking)}")
    
    # Compute features
    print("Computing features...")
    
    # Distance to goal
    shots_with_tracking['R'] = shots_with_tracking.apply(
        lambda row: compute_shot_to_goal_distance(row, event_to_tracking_map), axis=1
    )
    
    # Shooting angle (between goal posts)
    shots_with_tracking['shooting_angle'] = shots_with_tracking.apply(
        lambda row: compute_shooting_angle_from_cone(row, event_to_tracking_map), axis=1
    )
    
    # Goal angle (theta - angle to goal center from reference direction)
    shots_with_tracking['theta'] = shots_with_tracking.apply(
        lambda row: compute_goal_angle(row, event_to_tracking_map), axis=1
    )
    
    # PDI (simple cross product method)
    shots_with_tracking['PDI'] = shots_with_tracking.apply(
        lambda row: compute_ddi(row, event_to_tracking_map), axis=1
    )
    
    # Players in defensive cone (complete method)
    shots_with_tracking['PDI_complete'] = shots_with_tracking.apply(
        lambda row: count_players_in_cone(row, event_to_tracking_map), axis=1
    )
    
    # GK validation
    shots_with_tracking['gk_is_valid'] = shots_with_tracking.apply(
        lambda row: is_gk_in_target_area(row, event_to_tracking_map), axis=1
    )
    
    # Get valid GK frame
    shots_with_tracking['frame_gk'] = shots_with_tracking.apply(
        lambda row: get_frame_with_valid_gk(row, event_to_tracking_map), axis=1
    )
    
    # GK line offset
    shots_with_tracking['gk_line_offset'] = shots_with_tracking.apply(
        lambda row: compute_gk_line_offset(row, event_to_tracking_map), axis=1
    )
    
    # Voronoi area
    #shots_with_tracking['shooter_voronoi_area'] = shots_with_tracking.apply(
    #    lambda row: compute_shooter_voronoi_area(row, event_to_tracking_map), axis=1
    #)
    
    print(f"✓ Raw data merged and features computed: {len(shots_with_tracking)} records")
    return shots_with_tracking


def preprocess_merged_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the merged dataset with feature engineering and cleaning.
    Includes both original shot features and new tracking-based features.
    """
    print("Starting preprocessing of merged data...")
    
    # Check if we have tracking features (new pipeline) or need to do original preprocessing
    has_tracking_features = 'R' in merged_df.columns
    
    if not has_tracking_features:
        # Original preprocessing logic for backward compatibility
        from sklearn.preprocessing import MultiLabelBinarizer
        
        # Validate required columns exist for original preprocessing
        required_source_columns = [
            'id', 'matchId', 'matchTimestamp', 'videoTimestamp', 
            'location_x', 'location_y', 'team_id', 'player_id',
            'shot_bodyPart', 'assist_info_type_secondary', 'possession_types'
        ]
        
        missing_cols = [col for col in required_source_columns if col not in merged_df.columns]
        if missing_cols:
            print(f"⚠ Warning: Missing columns for original preprocessing: {missing_cols}")
            # Continue with available columns
        
        # One-hot encode shot_bodyPart if available
        if 'shot_bodyPart' in merged_df.columns:
            df_bodypart = pd.get_dummies(merged_df['shot_bodyPart'], prefix='shot_bodyPart')
            merged_df = pd.concat([merged_df, df_bodypart], axis=1)
            
            # Create shot_bodyPart_head_or_other from the one-hot encoded columns
            head_cols = [col for col in df_bodypart.columns if 'head' in col.lower() or 'other' in col.lower()]
            if head_cols:
                merged_df['shot_bodyPart_head_or_other'] = merged_df[head_cols].max(axis=1).astype(int)
            else:
                merged_df['shot_bodyPart_head_or_other'] = 0
        
        # One-hot encode assist_info_type_secondary if available
        if 'assist_info_type_secondary' in merged_df.columns:
            merged_df['assist_info_type_secondary'] = merged_df['assist_info_type_secondary'].fillna('').apply(
                lambda x: x if isinstance(x, list) else []
            )
            
            mlb_assist = MultiLabelBinarizer()
            assist_encoded = mlb_assist.fit_transform(merged_df['assist_info_type_secondary'])
            assist_df = pd.DataFrame(assist_encoded, 
                                    columns=[f"assist_type_{col}" for col in mlb_assist.classes_],
                                    index=merged_df.index)
            merged_df = pd.concat([merged_df, assist_df], axis=1)
            
            # Find and extract assist type columns
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
        
        # One-hot encode possession_types if available
        if 'possession_types' in merged_df.columns:
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

        # Select features for original pipeline
        original_features = [
            'id', 'matchId', 'matchTimestamp', 'videoTimestamp',
            'location_x', 'location_y', 'team_id', 'player_id',
            'shot_bodyPart_head_or_other', 'assist_type_long_pass', 
            'direct_free_kick', 'assist_type_through_pass'
        ]
        
        # Use only available columns
        available_original_features = [col for col in original_features if col in merged_df.columns]
        final_df = merged_df[available_original_features].copy()
        
        # Data type conversions for original features
        numeric_cols = ['location_x', 'location_y']
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        id_cols = ['id', 'matchId', 'team_id', 'player_id']
        for col in id_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str)
        
        binary_cols = ['shot_bodyPart_head_or_other', 'assist_type_long_pass', 
                       'direct_free_kick', 'assist_type_through_pass']
        for col in binary_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(int)
    
    else:
        # New pipeline with tracking features - keep both original and tracking features
        print("Detected tracking features - using enhanced preprocessing")
        
        # Define all possible feature columns (original + tracking)
        all_feature_columns = [
            # Original features
            'id', 'matchId', 'videoTimestamp', 'location_x', 'location_y', 
            'team_id', 'player_id',
            
            # Original derived features (if they exist)
            'shot_bodyPart_head_or_other', 'assist_type_long_pass', 
            'direct_free_kick', 'assist_type_through_pass',
            
            # New tracking-based features
            'R', 'shooting_angle', 'theta', 'PDI', 'PDI_complete',
            'gk_line_offset', 'shooter_voronoi_area'
        ]
        
        # Add outcome if available
        if 'outcome' in merged_df.columns:
            all_feature_columns.append('outcome')
        
        # Use only available columns
        available_columns = [col for col in all_feature_columns if col in merged_df.columns]
        final_df = merged_df[available_columns].copy()
        
        # Data type conversions
        numeric_cols = ['location_x', 'location_y', 'R', 'shooting_angle', 'theta', 'PDI', 'PDI_complete', 
                       'gk_line_offset', 'shooter_voronoi_area']
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        # Ensure ID columns are strings
        id_cols = ['id', 'matchId', 'team_id', 'player_id']
        for col in id_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str)
        
        # Handle binary features if they exist
        binary_cols = ['shot_bodyPart_head_or_other', 'assist_type_long_pass', 
                       'direct_free_kick', 'assist_type_through_pass']
        for col in binary_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(int)
    
    # Common handling for both pipelines
    
    # Handle missing values - drop rows with missing critical features
    critical_features = ['location_x', 'location_y']
    if has_tracking_features:
        critical_features.extend(['R', 'shooting_angle'])
    
    available_critical = [col for col in critical_features if col in final_df.columns]
    initial_rows = len(final_df)
    final_df = final_df.dropna(subset=available_critical)
    dropped_rows = initial_rows - len(final_df)
    
    if dropped_rows > 0:
        print(f"⚠ Dropped {dropped_rows} rows with missing critical features")
    
    # Fill missing timestamps with empty string if needed
    timestamp_cols = ['matchTimestamp', 'videoTimestamp']
    for col in timestamp_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna('')
    
    print(f"✓ Merged data preprocessed: {len(final_df)} final records")
    print(f"✓ Available features: {list(final_df.columns)}")
    return final_df


def preprocess_all_data(shots_dir: str, tracking_dir: str, mappings_dir: str,
                       prev_frames: int = 10, next_frames: int = 10) -> pd.DataFrame:
    """
    Complete preprocessing pipeline: merge and preprocess.
    """
    print("Starting data preprocessing pipeline...")
    
    # Step 1: Merge all raw data
    merged_df = merge_raw_data(shots_dir, tracking_dir, mappings_dir, prev_frames, next_frames)
    
    # Step 2: Preprocess the merged dataset
    final_df = preprocess_merged_data(merged_df)
    
    print("✓ Data preprocessing pipeline completed!")
    return final_df