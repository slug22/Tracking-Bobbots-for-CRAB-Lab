# smooth.py - Updated to use globals module
import globals as g  # Import globals module
import numpy as np

# Smoothing parameters
SMOOTHING_WINDOW = 30  # Number of frames to average over
POSITION_SMOOTHING_FACTOR = 0.6 # Lower = more smoothing (0.1-0.5 range)
WHITEST_SMOOTHING_FACTOR = 1.5   # Smoothing factor specifically for whitest points

def smooth_position(current_pos, previous_pos, alpha=POSITION_SMOOTHING_FACTOR):
    """Apply exponential moving average smoothing to position"""
    if previous_pos is None:
        return current_pos
    
    smoothed_x = alpha * current_pos[0] + (1 - alpha) * previous_pos[0]
    smoothed_y = alpha * current_pos[1] + (1 - alpha) * previous_pos[1]
    return (smoothed_x, smoothed_y)

def smooth_whitest_position(current_pos, previous_pos, alpha=WHITEST_SMOOTHING_FACTOR):
    """Apply exponential moving average smoothing specifically to whitest positions"""
    if previous_pos is None or current_pos is None:
        return current_pos
    
    smoothed_x = alpha * current_pos[0] + (1 - alpha) * previous_pos[0]
    smoothed_y = alpha * current_pos[1] + (1 - alpha) * previous_pos[1]
    return (smoothed_x, smoothed_y)

def apply_kalman_like_filter(robot_id, new_position, previous_position):
    """Simple Kalman-like filter for individual robot positions"""
    if previous_position is None:
        return new_position
    
    # Simple prediction (assume constant velocity)
    # NOW ACCESSING robot_paths THROUGH GLOBALS MODULE
    if robot_id in g.robot_paths and len(g.robot_paths[robot_id]) >= 2:
        # Calculate velocity from last two positions
        last_pos = g.robot_paths[robot_id][-1]
        second_last_pos = g.robot_paths[robot_id][-2]
        velocity_x = last_pos[0] - second_last_pos[0]
        velocity_y = last_pos[1] - second_last_pos[1]
        
        # Predict next position
        predicted_x = previous_position[0] + velocity_x
        predicted_y = previous_position[1] + velocity_y
        predicted_position = (predicted_x, predicted_y)
    else:
        predicted_position = previous_position
    
    # Simple Kalman gain (adjust these values to tune filtering)
    process_noise = 2.0  # How much we trust the prediction
    measurement_noise = 5.0  # How much we trust the measurement
    
    kalman_gain = process_noise / (process_noise + measurement_noise)
    
    # Update position
    filtered_x = predicted_position[0] + kalman_gain * (new_position[0] - predicted_position[0])
    filtered_y = predicted_position[1] + kalman_gain * (new_position[1] - predicted_position[1])
    
    return (filtered_x, filtered_y)

def apply_kalman_like_filter_whitest(robot_id, new_whitest_pos, previous_whitest_pos):
    """Simple Kalman-like filter for whitest positions with different parameters"""
    if previous_whitest_pos is None or new_whitest_pos is None:
        return new_whitest_pos
    
    # Simple prediction for whitest position (assume less predictable movement)
    predicted_position = previous_whitest_pos  # Simple prediction
    
    # Different Kalman parameters for whitest positions (more responsive to changes)
    process_noise = 3.0  # Higher process noise for more responsive tracking
    measurement_noise = 4.0  # Moderate measurement noise
    
    kalman_gain = process_noise / (process_noise + measurement_noise)
    
    # Update position
    filtered_x = predicted_position[0] + kalman_gain * (new_whitest_pos[0] - predicted_position[0])
    filtered_y = predicted_position[1] + kalman_gain * (new_whitest_pos[1] - predicted_position[1])
    
    return (filtered_x, filtered_y)

def calculate_smoothed_centroid(positions, use_median=True):
    """Calculate centroid with optional median filtering for outlier rejection"""
    if not positions:
        return None, None
    
    if use_median and len(positions) >= 3:
        # Remove outliers using median filtering
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Calculate median and MAD (Median Absolute Deviation)
        median_x = np.median(x_coords)
        median_y = np.median(y_coords)
        
        mad_x = np.median([abs(x - median_x) for x in x_coords])
        mad_y = np.median([abs(y - median_y) for y in y_coords])
        
        # Filter outliers (positions more than 2 MAD away from median)
        threshold = 2.0
        filtered_positions = []
        for pos in positions:
            if (mad_x == 0 or abs(pos[0] - median_x) <= threshold * mad_x) and \
               (mad_y == 0 or abs(pos[1] - median_y) <= threshold * mad_y):
                filtered_positions.append(pos)
        
        # Use filtered positions if we have enough, otherwise use all
        if len(filtered_positions) >= len(positions) * 0.5:  # Keep at least 50%
            positions = filtered_positions
    
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    
    return centroid_x, centroid_y