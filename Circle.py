# Circle.py - Updated to use globals module
import numpy as np
import cv2 as cv
from collections import defaultdict, deque
import threading
import queue
import time
import csv
import os
from math import sqrt, atan2
from smooth import smooth_position, smooth_whitest_position, apply_kalman_like_filter, apply_kalman_like_filter_whitest, calculate_smoothed_centroid
import globals as g  # IMPORT GLOBALS MODULE

# Remove all global variable declarations since they're now in globals.py
# The following lines are REMOVED:
# robot_positions = {}
# missing_robots = {}
# robot_paths = defaultdict(deque)
# etc...

def calculate_displacement(current_pos, initial_pos):
    """Calculate displacement from initial position"""
    if initial_pos is None:
        return 0
    return np.sqrt((current_pos[0] - initial_pos[0])**2 + (current_pos[1] - initial_pos[1])**2)

def calculate_smoothed_centroid_temporal(positions):
    """Calculate centroid with temporal smoothing using history"""
    # NOW USING g.centroid_history INSTEAD OF global centroid_history
    
    # Get current centroid
    current_centroid_x, current_centroid_y = calculate_smoothed_centroid(positions, use_median=True)
    
    if current_centroid_x is None or current_centroid_y is None:
        return None, None
    
    # Add to history - ACCESSING THROUGH GLOBALS MODULE
    g.centroid_history.append((current_centroid_x, current_centroid_y))
    
    # Apply temporal smoothing if we have enough history
    if len(g.centroid_history) >= 3:
        # Use weighted average with recent positions having more weight
        weights = np.linspace(0.5, 1.0, len(g.centroid_history))  # More recent = higher weight
        
        x_coords = [pos[0] for pos in g.centroid_history]
        y_coords = [pos[1] for pos in g.centroid_history]
        
        smoothed_x = np.average(x_coords, weights=weights)
        smoothed_y = np.average(y_coords, weights=weights)
        
        return smoothed_x, smoothed_y
    
    return current_centroid_x, current_centroid_y

def save_frame_data_to_csv(frame_number, timestamp, robot_positions, centroid_x, centroid_y, assigned_circles, dir_vec_x, dir_vec_y, angles, total_robots_for_centroid=None):
    """Save frame data to CSV format"""
    # Use total_robots_for_centroid if provided, otherwise use just active robots
    if total_robots_for_centroid is None:
        total_robots_for_centroid = len(robot_positions)
    
    frame_data = {
        'frame': frame_number,
        'timestamp': timestamp,
        'centroid_x': centroid_x if centroid_x is not None else '',
        'centroid_y': centroid_y if centroid_y is not None else '',
        'active_robots': total_robots_for_centroid  # Now includes missing robots in centroid calculation
    }
    
    # Add individual robot positions and directions
    for robot_id in range(1, g.MAX_ROBOTS + 1):  # USING g.MAX_ROBOTS
        if robot_id in robot_positions:
            frame_data[f'robot_{robot_id}_x'] = robot_positions[robot_id][0]
            frame_data[f'robot_{robot_id}_y'] = robot_positions[robot_id][1]
            
            # Find corresponding direction data
            robot_dir_x = 0
            robot_dir_y = 0
            robot_angle = 0
            
            # Match robot_id with assigned_circles to get direction index
            for i, (x, y, r, circle_robot_id) in enumerate(assigned_circles):
                if circle_robot_id == robot_id and i < len(dir_vec_x):
                    robot_angle = angles[i]
                    break
            
            frame_data[f'robot_{robot_id}_dir_x'] = robot_dir_x
            frame_data[f'robot_{robot_id}_dir_y'] = robot_dir_y
            frame_data[f'robot_{robot_id}_angle'] = robot_angle
        else:
            frame_data[f'robot_{robot_id}_x'] = ''
            frame_data[f'robot_{robot_id}_y'] = ''
            frame_data[f'robot_{robot_id}_dir_x'] = ''
            frame_data[f'robot_{robot_id}_dir_y'] = ''
            frame_data[f'robot_{robot_id}_angle'] = ''
    
    g.csv_data.append(frame_data)  # USING g.csv_data

def write_csv_file(filename):
    """Write all collected data to CSV file"""
    if not g.csv_data:  # USING g.csv_data
        print("No data to write to CSV")
        return
    
    # Prepare CSV headers
    headers = ['frame', 'timestamp', 'centroid_x', 'centroid_y', 'active_robots']
    
    # Add robot position and direction headers
    for robot_id in range(1, g.MAX_ROBOTS + 1):  # USING g.MAX_ROBOTS
        headers.extend([f'robot_{robot_id}_x', f'robot_{robot_id}_y', 
                       f'robot_{robot_id}_dir_x', f'robot_{robot_id}_dir_y', 
                       f'robot_{robot_id}_angle'])
    
    # Write to CSV file
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(g.csv_data)  # USING g.csv_data
        print(f"Data successfully written to {filename}")
        print(f"Total frames recorded: {len(g.csv_data)}")  # USING g.csv_data
    except Exception as e:
        print(f"Error writing CSV file: {e}")

def assign_robot_ids(circles, robot_positions, missing_robots, next_robot_id, frame=None):
    """
    Assign IDs to detected circles based on proximity to previous positions
    Also handles reassigning IDs to robots that went missing
    Now includes proper per-robot whitest pixel tracking
    """
    # NO MORE global robot_whitest_history - now using g.robot_whitest_history
    
    if circles is None:
        # Update missing robot frame counts
        robots_to_remove = []
        for robot_id in missing_robots:
            missing_robots[robot_id]['missing_frames'] += 1
            if missing_robots[robot_id]['missing_frames'] > g.MAX_MISSING_FRAMES:  # USING g.MAX_MISSING_FRAMES
                robots_to_remove.append(robot_id)
        
        # Remove robots that have been missing too long
        for robot_id in robots_to_remove:
            del missing_robots[robot_id]
            # Clean up whitest history for removed robots
            if robot_id in g.robot_whitest_history:  # USING g.robot_whitest_history
                del g.robot_whitest_history[robot_id]
        
        return robot_positions, missing_robots, next_robot_id, [], []
    
    current_circles = []
    new_robot_positions = {}
    assigned_ids = []
    
    # Convert frame to grayscale if it's provided and in color
    if frame is not None:
        if len(frame.shape) == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
    
    rad = 24
    
    # First, detect all circles and find their whitest pixels
    circle_whitest_data = []  # Store (x, y, rad, whitest_x, whitest_y, max_value) for each circle
    
    for circle in circles[0, :]:
        x, y = int(circle[0]), int(circle[1])
        current_circles.append((x, y, rad))
        
        # Find whitest pixel in annular region if frame is provided
        whitest_pos = None
        max_value = -1
        
        if frame is not None:
            inner_radius = round(rad * 0.4)
            outer_radius = round(rad * 0.5)
            
            # Define bounding box for efficiency
            x_min = max(0, x - outer_radius)
            x_max = min(gray_frame.shape[1], x + outer_radius + 1)
            y_min = max(0, y - outer_radius)
            y_max = min(gray_frame.shape[0], y + outer_radius + 1)
            
            # Search within bounding box
            for py in range(y_min, y_max):
                for px in range(x_min, x_max):
                    distance = np.sqrt((px - x)**2 + (py - y)**2)
                    
                    if inner_radius <= distance <= outer_radius:
                        pixel_value = gray_frame[py, px]
                        if pixel_value > max_value:
                            max_value = pixel_value
                            whitest_pos = (px, py)
        
        circle_whitest_data.append((x, y, rad, whitest_pos, max_value))
    
    # Now assign robot IDs (existing logic with small modification to track indices)
    unmatched_circles = list(enumerate(current_circles))  # Keep track of indices
    
    # First, try to match with currently active robots
    for robot_id, (prev_x, prev_y) in robot_positions.items():
        best_match = None
        min_distance = float('inf')
        
        for i, (circle_idx, (x, y, r)) in enumerate(unmatched_circles):
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            if distance < g.TRACKING_DISTANCE_THRESHOLD and distance < min_distance:  # USING g.TRACKING_DISTANCE_THRESHOLD
                min_distance = distance
                best_match = i
        
        if best_match is not None:
            circle_idx, (x, y, r) = unmatched_circles[best_match]
            new_robot_positions[robot_id] = (x, y)
            assigned_ids.append((x, y, r, robot_id, circle_idx))  # Include circle_idx for whitest tracking
            unmatched_circles.pop(best_match)
    
    # Move robots that weren't matched to missing list
    for robot_id in robot_positions:
        if robot_id not in new_robot_positions:
            if robot_id not in missing_robots:
                # Store both position and last known whitest position
                last_whitest_pos = g.previous_smoothed_whitest.get(robot_id, None)  # USING g.previous_smoothed_whitest
                missing_robots[robot_id] = {
                    'last_pos': robot_positions[robot_id],
                    'last_whitest_pos': last_whitest_pos,  # NEW: Store last known whitest position
                    'missing_frames': 1
                }
            else:
                missing_robots[robot_id]['missing_frames'] += 1
    
    # Try to match unmatched circles with missing robots
    circles_to_remove = []
    for i, (circle_idx, (x, y, r)) in enumerate(unmatched_circles):
        best_missing_match = None
        min_missing_distance = float('inf')
        
        for robot_id, info in missing_robots.items():
            last_x, last_y = info['last_pos']
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if distance < g.MISSING_ROBOT_THRESHOLD and distance < min_missing_distance:  # USING g.MISSING_ROBOT_THRESHOLD
                min_missing_distance = distance
                best_missing_match = robot_id
        
        if best_missing_match is not None:
            # Reassign the missing robot ID and restore last known whitest position
            new_robot_positions[best_missing_match] = (x, y)
            assigned_ids.append((x, y, r, best_missing_match, circle_idx))
            circles_to_remove.append(i)
            
            # Restore the last known whitest position for this robot
            if 'last_whitest_pos' in missing_robots[best_missing_match] and missing_robots[best_missing_match]['last_whitest_pos'] is not None:
                g.previous_smoothed_whitest[best_missing_match] = missing_robots[best_missing_match]['last_whitest_pos']  # USING g.previous_smoothed_whitest
                print(f"Robot {best_missing_match}: Restored last known whitest position {missing_robots[best_missing_match]['last_whitest_pos']}")
            
            # Remove from missing list
            del missing_robots[best_missing_match]
    
    # Remove matched circles from unmatched list (in reverse order to maintain indices)
    for i in sorted(circles_to_remove, reverse=True):
        unmatched_circles.pop(i)
    
    # Assign new IDs to remaining unmatched circles (completely new robots)
    for circle_idx, (x, y, r) in unmatched_circles:
        if len(new_robot_positions) + len(missing_robots) < g.MAX_ROBOTS and next_robot_id <= g.MAX_ROBOTS:  # USING g.MAX_ROBOTS
            new_robot_positions[next_robot_id] = (x, y)
            assigned_ids.append((x, y, r, next_robot_id, circle_idx))
            next_robot_id += 1
    
    # Now process whitest positions with proper per-robot tracking
    whitest_positions = []
    
    for x, y, r, robot_id, circle_idx in assigned_ids:
        circle_x, circle_y, circle_rad, whitest_pos, max_value = circle_whitest_data[circle_idx]
        
        final_whitest_x = None
        final_whitest_y = None
        final_max_value = None
        
        if whitest_pos is not None:
            whitest_x, whitest_y = whitest_pos
            
            # Check if this robot has previous whitest position history
            if robot_id in g.robot_whitest_history and len(g.robot_whitest_history[robot_id]) > 0:  # USING g.robot_whitest_history
                last_whitest_x, last_whitest_y = g.robot_whitest_history[robot_id][-1]
                
                # Calculate distance from last known whitest position for THIS robot
                whitest_distance = sqrt((whitest_x - last_whitest_x)**2 + (whitest_y - last_whitest_y)**2)
                
                if whitest_distance < g.WHITEST_DISTANCE_THRESHOLD:  # USING g.WHITEST_DISTANCE_THRESHOLD
                    # Whitest position is within acceptable range, use it
                    final_whitest_x = whitest_x
                    final_whitest_y = whitest_y
                    final_max_value = max_value
                    # Update history for this robot
                    g.robot_whitest_history[robot_id].append((whitest_x, whitest_y))  # USING g.robot_whitest_history
                else:
                    # Whitest position jumped too far, check if last known position is still within current circle
                    last_distance_from_center = sqrt((last_whitest_x - circle_x)**2 + (last_whitest_y - circle_y)**2)
                    circle_radius = 28  # Use actual circle radius for bounds checking
                    
                    if last_distance_from_center <= circle_radius:
                        # Last known position is still within current circle bounds, use it
                        final_whitest_x = last_whitest_x
                        final_whitest_y = last_whitest_y
                        final_max_value = None  # Indicate this is a carried-over position
                        print(f"Robot {robot_id}: Whitest position jumped {whitest_distance:.1f} pixels, using last known position")
                    else:
                        # Last known position is outside current circle, use new detected position
                        final_whitest_x = whitest_x
                        final_whitest_y = whitest_y
                        final_max_value = max_value
                        # Update history for this robot
                        g.robot_whitest_history[robot_id].append((whitest_x, whitest_y))  # USING g.robot_whitest_history
                        print(f"Robot {robot_id}: Last known position outside circle bounds, accepting new position")
            else:
                # First time seeing this robot, accept the whitest position
                final_whitest_x = whitest_x
                final_whitest_y = whitest_y
                final_max_value = max_value
                # Initialize history for this robot
                g.robot_whitest_history[robot_id].append((whitest_x, whitest_y))  # USING g.robot_whitest_history
        else:
            # No whitest position found, use last known if available for this robot AND within current circle
            if robot_id in g.robot_whitest_history and len(g.robot_whitest_history[robot_id]) > 0:  # USING g.robot_whitest_history
                last_whitest_x, last_whitest_y = g.robot_whitest_history[robot_id][-1]
                last_distance_from_center = sqrt((last_whitest_x - circle_x)**2 + (last_whitest_y - circle_y)**2)
                circle_radius = 28  # Use actual circle radius for bounds checking
                
                if last_distance_from_center <= circle_radius:
                    # Last known position is within current circle bounds
                    final_whitest_x, final_whitest_y = last_whitest_x, last_whitest_y
                    final_max_value = None
                # If last known position is outside bounds, leave final_whitest as None
        
        # Limit history length for this robot
        if robot_id in g.robot_whitest_history and len(g.robot_whitest_history[robot_id]) > g.MAX_WHITEST_HISTORY:  # USING g.robot_whitest_history and g.MAX_WHITEST_HISTORY
            g.robot_whitest_history[robot_id].popleft()
        
        whitest_positions.append((x, y, r, final_whitest_x, final_whitest_y, final_max_value))
    
    # Clean up missing robots that have been missing too long
    robots_to_remove = []
    for robot_id in missing_robots:
        if missing_robots[robot_id]['missing_frames'] > g.MAX_MISSING_FRAMES:  # USING g.MAX_MISSING_FRAMES
            robots_to_remove.append(robot_id)
    
    for robot_id in robots_to_remove:
        del missing_robots[robot_id]
        if robot_id in g.robot_whitest_history:  # USING g.robot_whitest_history
            del g.robot_whitest_history[robot_id]
    
    # Convert assigned_ids back to original format (remove circle_idx)
    assigned_ids_clean = [(x, y, r, robot_id) for x, y, r, robot_id, circle_idx in assigned_ids]
    
    return new_robot_positions, missing_robots, next_robot_id, assigned_ids_clean, whitest_positions

def assign_robot_ids_smoothed(circles, robot_positions, missing_robots, next_robot_id, frame=None):
    """Enhanced version with position smoothing for both circles and whitest points"""
    # NO MORE global declarations - using g.* for all globals
    
    if circles is None:
        # Same missing robot handling as before...
        robots_to_remove = []
        for robot_id in missing_robots:
            missing_robots[robot_id]['missing_frames'] += 1
            if missing_robots[robot_id]['missing_frames'] > g.MAX_MISSING_FRAMES:  # USING g.MAX_MISSING_FRAMES
                robots_to_remove.append(robot_id)
        
        for robot_id in robots_to_remove:
            del missing_robots[robot_id]
            if robot_id in g.robot_whitest_history:  # USING g.robot_whitest_history
                del g.robot_whitest_history[robot_id]
            if robot_id in g.robot_position_filters:  # USING g.robot_position_filters
                del g.robot_position_filters[robot_id]
            if robot_id in g.previous_smoothed_positions:  # USING g.previous_smoothed_positions
                del g.previous_smoothed_positions[robot_id]
            if robot_id in g.robot_whitest_filters:  # USING g.robot_whitest_filters
                del g.robot_whitest_filters[robot_id]
            if robot_id in g.previous_smoothed_whitest:  # USING g.previous_smoothed_whitest
                del g.previous_smoothed_whitest[robot_id]
        
        return robot_positions, missing_robots, next_robot_id, [], []
    
    # Get raw positions first (use existing logic)
    raw_positions, missing_robots, next_robot_id, assigned_ids, raw_whitest_positions = assign_robot_ids(
        circles, robot_positions, missing_robots, next_robot_id, frame)
    
    # Apply smoothing to robot circle positions
    smoothed_positions = {}
    
    for robot_id, raw_pos in raw_positions.items():
        # Get previous smoothed position
        prev_smoothed_pos = g.previous_smoothed_positions.get(robot_id)  # USING g.previous_smoothed_positions
        
        # Apply Kalman-like filtering
        filtered_pos = apply_kalman_like_filter(robot_id, raw_pos, prev_smoothed_pos)
        
        # Apply exponential smoothing
        smoothed_pos = smooth_position(filtered_pos, prev_smoothed_pos)
        
        smoothed_positions[robot_id] = smoothed_pos
        g.previous_smoothed_positions[robot_id] = smoothed_pos  # USING g.previous_smoothed_positions
        
        # Update position history for this robot
        g.robot_position_filters[robot_id]['x'].append(smoothed_pos[0])  # USING g.robot_position_filters
        g.robot_position_filters[robot_id]['y'].append(smoothed_pos[1])
        
        # Update robot paths with smoothed positions
        g.robot_paths[robot_id].append(smoothed_pos)  # USING g.robot_paths
        if len(g.robot_paths[robot_id]) > g.MAX_PATH_LENGTH:  # USING g.MAX_PATH_LENGTH
            g.robot_paths[robot_id].popleft()
    
    # NEW: Apply smoothing to whitest positions
    smoothed_whitest_positions = []
    
    for i, (x, y, r, whitest_x, whitest_y, max_value) in enumerate(raw_whitest_positions):
        # Find the corresponding robot_id for this whitest position
        robot_id = None
        smoothed_circle_x, smoothed_circle_y = x, y  # Default to raw if no match found
        
        # Match with assigned_ids to get robot_id and smoothed circle position
        for smooth_x, smooth_y, smooth_r, rid in assigned_ids:
            if abs(smooth_x - x) < 30 and abs(smooth_y - y) < 30:  # Close match
                robot_id = rid
                smoothed_circle_x, smoothed_circle_y = smoothed_positions[rid]
                break
        
        # Initialize smoothed whitest position
        smoothed_whitest_x, smoothed_whitest_y = whitest_x, whitest_y
        
        if robot_id is not None:
            # Get previous smoothed whitest position for this robot
            prev_smoothed_whitest_pos = g.previous_smoothed_whitest.get(robot_id)  # USING g.previous_smoothed_whitest
            
            if whitest_x is not None and whitest_y is not None:
                # Current whitest point detected - apply smoothing
                
                # Apply Kalman-like filtering to whitest position
                filtered_whitest_pos = apply_kalman_like_filter_whitest(
                    robot_id, (whitest_x, whitest_y), prev_smoothed_whitest_pos)
                
                # Apply exponential smoothing to whitest position
                smoothed_whitest_pos = smooth_whitest_position(
                    filtered_whitest_pos, prev_smoothed_whitest_pos)
                
                if smoothed_whitest_pos is not None:
                    smoothed_whitest_x, smoothed_whitest_y = smoothed_whitest_pos
                    g.previous_smoothed_whitest[robot_id] = smoothed_whitest_pos  # USING g.previous_smoothed_whitest
                    
                    # Update whitest position history for this robot
                    g.robot_whitest_filters[robot_id]['x'].append(smoothed_whitest_x)  # USING g.robot_whitest_filters
                    g.robot_whitest_filters[robot_id]['y'].append(smoothed_whitest_y)
                    
            else:
                # Current whitest point is missing - use last known smoothed position
                if prev_smoothed_whitest_pos is not None:
                    smoothed_whitest_x, smoothed_whitest_y = prev_smoothed_whitest_pos
                    print(f"Robot {robot_id}: Whitest point missing, using last known smoothed position ({smoothed_whitest_x:.1f}, {smoothed_whitest_y:.1f})")
                    
                    # Verify the last known position is still within reasonable bounds of current circle
                    distance_from_center = sqrt((smoothed_whitest_x - smoothed_circle_x)**2 + (smoothed_whitest_y - smoothed_circle_y)**2)
                    circle_radius = 30  # Slightly larger radius for bounds checking
                    
                    if distance_from_center > circle_radius:
                        # Last known position is too far from current circle, set to None
                        smoothed_whitest_x, smoothed_whitest_y = None, None
                        print(f"Robot {robot_id}: Last known whitest position too far from current circle ({distance_from_center:.1f} > {circle_radius}), discarding")
                else:
                    # No previous position available
                    smoothed_whitest_x, smoothed_whitest_y = None, None
        
        smoothed_whitest_positions.append((smoothed_circle_x, smoothed_circle_y, r, 
                                         smoothed_whitest_x, smoothed_whitest_y, max_value, robot_id))
    
    # Update assigned_ids with smoothed circle positions
    smoothed_assigned_ids = []
    for x, y, r, robot_id in assigned_ids:
        if robot_id in smoothed_positions:
            smooth_x, smooth_y = smoothed_positions[robot_id]
            smoothed_assigned_ids.append((smooth_x, smooth_y, r, robot_id))
        else:
            smoothed_assigned_ids.append((x, y, r, robot_id))
    
    return smoothed_positions, missing_robots, next_robot_id, smoothed_assigned_ids, smoothed_whitest_positions

def draw_arrow(img, start, end, color, thickness=2, arrow_length=10):
    """Draw an arrow from start point to end point"""
    if start is None or end is None:
        return
    
    start_point = (int(start[0]), int(start[1]))
    end_point = (int(end[0]), int(end[1]))
    
    # Draw the line
    cv.line(img, start_point, end_point, color, thickness)
    
    # Calculate arrow head
    angle = atan2(end[1] - start[1], end[0] - start[0])
    
    # Arrow head points
    arrow_head1 = (
        int(end[0] - arrow_length * np.cos(angle - np.pi/6)),
        int(end[1] - arrow_length * np.sin(angle - np.pi/6))
    )
    arrow_head2 = (
        int(end[0] - arrow_length * np.cos(angle + np.pi/6)),
        int(end[1] - arrow_length * np.sin(angle + np.pi/6))
    )
    
    # Draw arrow head
    cv.line(img, end_point, arrow_head1, color, thickness)
    cv.line(img, end_point, arrow_head2, color, thickness)

def video_processing(vid_filename):
    """Main video processing function with CSV data collection"""
    # NO MORE global declarations - using g.reset_globals() instead
    
    # Reset all tracking variables for each new video using globals module
    g.reset_globals()
    
    # Open video file or camera (0 for webcam)
    cap = cv.VideoCapture(vid_filename)  # or use 0 for webcam
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Store initial positions for displacement calculation
    robot_initial_positions = {}
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or failed to read frame")
                break
            
            frame = cv.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
            frame = frame[200:3000, 1150:2200]
            
            frame_count += 1
            current_time = time.time() - start_time
            
            # Convert to grayscale
            img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            img = cv.medianBlur(img, 5)
            
            # Circle detection
            circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, 
                                     param1=50, param2=30, minRadius=15, maxRadius=34)
            
            # Updated call to assign_robot_ids_smoothed instead of assign_robot_ids
            # NOW USING GLOBALS MODULE FOR ALL VARIABLES
            g.robot_positions, g.missing_robots, g.next_robot_id, assigned_circles, smoothed_whitest_positions = assign_robot_ids_smoothed(
                circles, g.robot_positions, g.missing_robots, g.next_robot_id, img)

            # Calculate direction vectors using smoothed positions
            dir_vec_x = [0 for _ in range(len(smoothed_whitest_positions))]
            dir_vec_y = [0 for _ in range(len(smoothed_whitest_positions))]
            angles = [0 for _ in range(len(smoothed_whitest_positions))]

            for i, (smoothed_circle_x, smoothed_circle_y, r, smoothed_whitest_x, smoothed_whitest_y, max_value, robot_id) in enumerate(smoothed_whitest_positions):
                if smoothed_whitest_x is not None and smoothed_whitest_y is not None:
                    # Use smoothed positions for direction calculation
                    x_vec = smoothed_whitest_x - smoothed_circle_x
                    y_vec = smoothed_whitest_y - smoothed_circle_y
                    dist = sqrt(x_vec**2 + y_vec**2)
                    
                    if dist > 0:
                        dir_vec_x[i] = x_vec/dist
                        dir_vec_y[i] = y_vec/dist
                        angles[i] = atan2(x_vec, y_vec)

            # Calculate centroid using all robot positions (active smoothed + missing last smoothed)
            all_robot_positions = []
            
            # Add active robots' current smoothed positions
            for robot_id in g.robot_positions:  # USING g.robot_positions
                all_robot_positions.append(g.robot_positions[robot_id])
            
            # Add missing robots' last known smoothed positions
            for robot_id, info in g.missing_robots.items():  # USING g.missing_robots
                all_robot_positions.append(info['last_pos'])
            
            # Simple centroid calculation - no smoothing, just average of all positions
            if all_robot_positions:
                centroid_x = sum(pos[0] for pos in all_robot_positions) / len(all_robot_positions)
                centroid_y = sum(pos[1] for pos in all_robot_positions) / len(all_robot_positions)
            else:
                centroid_x, centroid_y = None, None
            
            # Calculate total robots for display and CSV (active + missing)
            total_robots_for_centroid = len(all_robot_positions)
            
            # Save frame data to CSV storage
            save_frame_data_to_csv(frame_count, current_time, g.robot_positions, centroid_x, centroid_y, assigned_circles, dir_vec_x, dir_vec_y, angles, total_robots_for_centroid)            
            
            # Draw circles with IDs on the original color frame (using smoothed positions)
            for x, y, r, robot_id in assigned_circles:
                # Use a simple color scheme (cycling through basic colors)
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                         (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
                         (0, 128, 0), (128, 128, 0)]
                color = colors[(robot_id - 1) % len(colors)]
                
                # Draw the outer circle
                cv.circle(frame, (int(x), int(y)), 30, color, 2)
                cv.circle(frame, (int(x), int(y)), round(30 * .55), color, 2)
                # Draw the center of the circle
                cv.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)
                # Draw the robot ID
                cv.putText(frame, f'R{robot_id}', (int(x)-10, int(y)-28-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw whitest pixels and arrows using smoothed positions
            FIXED_ARROW_LENGTH = 30  # Set your desired arrow length here
            
            for i, (circle_x, circle_y, radius, whitest_x, whitest_y, pixel_value, robot_id) in enumerate(smoothed_whitest_positions):
                if whitest_x is not None and whitest_y is not None:
                    # Get the color for this robot
                    if robot_id is not None:
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
                                 (0, 128, 0), (128, 128, 0)]
                        color = colors[(robot_id - 1) % len(colors)]
                    else:
                        color = (255, 255, 255)  # Default white if no robot ID found
                    
                    # Calculate direction vector using smoothed positions
                    dx = whitest_x - circle_x
                    dy = whitest_y - circle_y
                    dist = sqrt(dx**2 + dy**2)
                    
                    # Only draw arrow if there's a valid direction
                    if dist > 0.1:  # Avoid division by very small numbers
                        # Normalize direction and scale to fixed length
                        unit_dx = dx / dist
                        unit_dy = dy / dist
                        
                        # Calculate fixed-length arrow end point
                        arrow_end_x = circle_x + unit_dx * FIXED_ARROW_LENGTH
                        arrow_end_y = circle_y + unit_dy * FIXED_ARROW_LENGTH
                        
                        # Draw fixed-length arrow using smoothed positions
                        draw_arrow(frame, (circle_x, circle_y), (arrow_end_x, arrow_end_y), color, thickness=2, arrow_length=8)
                    
                    # Draw a small marker at the smoothed whitest pixel location
                    cv.circle(frame, (int(whitest_x), int(whitest_y)), 3, color, -1)
                    
                    # Optionally display the pixel value
                    if pixel_value is not None:
                        cv.putText(frame, f'{pixel_value}', (int(whitest_x)+5, int(whitest_y)-5), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw centroid if available
            if centroid_x is not None and centroid_y is not None:
                cv.circle(frame, (int(centroid_x), int(centroid_y)), 8, (255, 255, 255), -1)
                cv.circle(frame, (int(centroid_x), int(centroid_y)), 10, (0, 0, 0), 2)
                cv.putText(frame, 'CENTROID', (int(centroid_x)-30, int(centroid_y)-15), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw centroid history trail for visualization
            if len(g.centroid_history) > 1:  # USING g.centroid_history
                for i in range(1, len(g.centroid_history)):
                    prev_x, prev_y = g.centroid_history[i-1]
                    curr_x, curr_y = g.centroid_history[i]
                    # Draw line with fading color
                    alpha = i / len(g.centroid_history)  # Fade from old to new
                    color_intensity = int(255 * alpha)
                    cv.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), 
                           (color_intensity, color_intensity, color_intensity), 1)
            
            # Draw missing robot positions using smoothed last known positions
            for robot_id, info in g.missing_robots.items():  # USING g.missing_robots
                last_x, last_y = info['last_pos']
                missing_frames = info['missing_frames']
                
                # Get the color for this robot
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                         (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
                         (0, 128, 0), (128, 128, 0)]
                color = colors[(robot_id - 1) % len(colors)]
                
                # Draw the robot exactly like active robots (same opacity)
                # Draw the outer circle
                cv.circle(frame, (int(last_x), int(last_y)), 30, color, 2)
                cv.circle(frame, (int(last_x), int(last_y)), round(30 * .55), color, 2)
                # Draw the center of the circle
                cv.circle(frame, (int(last_x), int(last_y)), 2, (0, 0, 255), 3)
                # Draw the robot ID
                cv.putText(frame, f'R{robot_id}', (int(last_x)-10, int(last_y)-28-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw last known arrow using stored whitest position from missing_robots info
                if 'last_whitest_pos' in info and info['last_whitest_pos'] is not None:
                    last_whitest_x, last_whitest_y = info['last_whitest_pos']
                    
                    # Calculate direction vector
                    dx = last_whitest_x - last_x
                    dy = last_whitest_y - last_y
                    dist = sqrt(dx**2 + dy**2)
                    
                    if dist > 0.1:  # Only draw if valid direction
                        # Normalize direction and scale to fixed length
                        unit_dx = dx / dist
                        unit_dy = dy / dist
                        
                        # Calculate fixed-length arrow end point
                        arrow_end_x = last_x + unit_dx * FIXED_ARROW_LENGTH
                        arrow_end_y = last_y + unit_dy * FIXED_ARROW_LENGTH
                        
                        # Draw arrow from last known center to calculated end point
                        draw_arrow(frame, (last_x, last_y), (arrow_end_x, arrow_end_y), color, thickness=2, arrow_length=8)
                        
                        # Draw small marker at last known smoothed whitest pixel
                        cv.circle(frame, (int(last_whitest_x), int(last_whitest_y)), 3, color, -1)
                        
                        # Add text indicator that this is a preserved direction
                        cv.putText(frame, 'LAST', (int(last_whitest_x)+5, int(last_whitest_y)-8), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Display status information
            active_count = len(assigned_circles)
            missing_count = len(g.missing_robots)  # USING g.missing_robots
            total_tracked = active_count + missing_count
            
            cv.putText(frame, f'Active: {active_count}, Missing: {missing_count}, Total: {total_tracked}/10', 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, f'Frame: {frame_count}/{total_frames}', 
                      (10, height-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if centroid_x is not None and centroid_y is not None:
                cv.putText(frame, f'Centroid: ({int(centroid_x)}, {int(centroid_y)})', 
                          (10, height-40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv.imshow('Robot Tracking', frame)
            
            # Progress update every 100 frames
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # Press 'q' to quit early
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Processing interrupted by user")
                break
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv.destroyAllWindows()
        
        # Generate output filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f'csv_data/{vid_filename}.csv'
        
        # Write CSV data
        write_csv_file(csv_filename)
        
        print(f"Processing complete. Processed {frame_count} frames.")
        print(f"CSV data saved to: {csv_filename}")


if __name__ == "__main__":
    # Run the video processing
    folder_path = 'videos-use-this' # Replace with your folder's path
    i = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename) # Get full path
        i +=1
        if os.path.isfile(file_path) and i > 3:

            print(filename)
            video_processing(file_path)