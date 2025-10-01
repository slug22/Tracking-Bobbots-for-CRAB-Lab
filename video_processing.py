import numpy as np
import cv2 as cv
from collections import defaultdict, deque
import threading
import queue
import time
import csv
import os
from math import sqrt, atan2

def video_processing(vid_filename):
    """Main video processing function with CSV data collection"""
    global robot_positions, missing_robots, next_robot_id
    global robot_paths, robot_displacement_history, frame_timestamps, csv_data, robot_whitest_history
    global centroid_history, robot_position_filters, previous_smoothed_positions
    global robot_whitest_filters, previous_smoothed_whitest
    
    # Reset all tracking variables for each new video
    robot_positions = {}
    missing_robots = {}
    robot_whitest_history = defaultdict(deque)
    centroid_history = deque(maxlen=10)
    robot_position_filters = defaultdict(lambda: {'x': deque(maxlen=5), 'y': deque(maxlen=5)})
    previous_smoothed_positions = {}
    robot_whitest_filters = defaultdict(lambda: {'x': deque(maxlen=5), 'y': deque(maxlen=5)})
    previous_smoothed_whitest = {}
    next_robot_id = 1
    csv_data = []
    robot_paths = defaultdict(deque)
    robot_displacement_history = defaultdict(list)
    frame_timestamps = []
    
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
            robot_positions, missing_robots, next_robot_id, assigned_circles, smoothed_whitest_positions = assign_robot_ids_smoothed(
                circles, robot_positions, missing_robots, next_robot_id, img)

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
            for robot_id in robot_positions:
                all_robot_positions.append(robot_positions[robot_id])
            
            # Add missing robots' last known smoothed positions
            for robot_id, info in missing_robots.items():
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
            save_frame_data_to_csv(frame_count, current_time, robot_positions, centroid_x, centroid_y, assigned_circles, dir_vec_x, dir_vec_y, angles, total_robots_for_centroid)            
            
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
            if len(centroid_history) > 1:
                for i in range(1, len(centroid_history)):
                    prev_x, prev_y = centroid_history[i-1]
                    curr_x, curr_y = centroid_history[i]
                    # Draw line with fading color
                    alpha = i / len(centroid_history)  # Fade from old to new
                    color_intensity = int(255 * alpha)
                    cv.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), 
                           (color_intensity, color_intensity, color_intensity), 1)
            
            # Draw missing robot positions using smoothed last known positions
            for robot_id, info in missing_robots.items():
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
            missing_count = len(missing_robots)
            total_tracked = active_count + missing_count
            
            cv.putText(frame, f'Active: {active_count}, Missing: {missing_count}, Total: {total_tracked}/10', 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, f'Frame: {frame_count}/{total_frames}', 
                      (10, height-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if centroid_x is not None and centroid_y is not None:
                cv.putText(frame, f'Centroid: ({int(centroid_x)}, {int(centroid_y)})', 
                          (10, height-40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display smoothing status for both circles and whitest points
            cv.putText(frame, f'Circle Smoothing: α={POSITION_SMOOTHING_FACTOR}', 
                      (10, height-60), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv.putText(frame, f'Whitest Smoothing: α={WHITEST_SMOOTHING_FACTOR}', 
                      (10, height-80), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
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
        print(f"Smoothing applied:")
        print(f"  - Circle positions: α={POSITION_SMOOTHING_FACTOR}")
        print(f"  - Whitest positions: α={WHITEST_SMOOTHING_FACTOR}")

