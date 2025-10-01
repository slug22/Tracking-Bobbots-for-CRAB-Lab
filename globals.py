# globals.py
from collections import defaultdict, deque

# Global variables that need to be shared between modules
robot_positions = {}  # Dictionary to store current robot positions {id: (x, y)}
missing_robots = {}   # Dictionary to store missing robot info {id: {'last_pos': (x, y), 'missing_frames': count}}
robot_paths = defaultdict(deque)  # Store path history for each robot
robot_displacement_history = defaultdict(list)  # Store displacement over time
frame_timestamps = []  # Store frame timestamps

# Per-robot whitest position tracking
robot_whitest_history = defaultdict(deque)  # Store whitest position history per robot {robot_id: deque of (x, y)}

# Smoothing variables for both circles and whitest points
centroid_history = deque(maxlen=10)  # Store recent centroid positions
robot_position_filters = defaultdict(lambda: {'x': deque(maxlen=5), 'y': deque(maxlen=5)})
previous_smoothed_positions = {}  # Store previous smoothed positions for each robot

# Whitest point smoothing variables
robot_whitest_filters = defaultdict(lambda: {'x': deque(maxlen=5), 'y': deque(maxlen=5)})
previous_smoothed_whitest = {}  # Store previous smoothed whitest positions for each robot

next_robot_id = 1

# Constants
MAX_ROBOTS = 10
TRACKING_DISTANCE_THRESHOLD = 30  # Maximum distance to consider same robot
MISSING_ROBOT_THRESHOLD = 50    # Distance threshold for reassigning missing robot IDs
MAX_MISSING_FRAMES = 30          # Remove missing robot after this many frames
MAX_PATH_LENGTH = 100            # Maximum path points to store per robot
WHITEST_DISTANCE_THRESHOLD = 14  # Maximum distance for whitest position to be considered valid
MAX_WHITEST_HISTORY = 10         # Maximum number of whitest positions to store per robot

# CSV data storage
csv_data = []

def reset_globals():
    """Function to reset all global variables for a new video"""
    global robot_positions, missing_robots, robot_paths, robot_displacement_history
    global frame_timestamps, robot_whitest_history, centroid_history
    global robot_position_filters, previous_smoothed_positions
    global robot_whitest_filters, previous_smoothed_whitest, next_robot_id, csv_data
    
    robot_positions = {}
    missing_robots = {}
    robot_paths = defaultdict(deque)
    robot_displacement_history = defaultdict(list)
    frame_timestamps = []
    robot_whitest_history = defaultdict(deque)
    centroid_history = deque(maxlen=10)
    robot_position_filters = defaultdict(lambda: {'x': deque(maxlen=5), 'y': deque(maxlen=5)})
    previous_smoothed_positions = {}
    robot_whitest_filters = defaultdict(lambda: {'x': deque(maxlen=5), 'y': deque(maxlen=5)})
    previous_smoothed_whitest = {}
    next_robot_id = 1
    csv_data = []