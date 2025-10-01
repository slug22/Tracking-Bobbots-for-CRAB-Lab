import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import io

# Sample data from your CSV
data = """frame,timestamp,centroid_x,centroid_y,active_robots,robot_1_x,robot_1_y,robot_1_dir_x,robot_1_dir_y,robot_1_angle,robot_2_x,robot_2_y,robot_2_dir_x,robot_2_dir_y,robot_2_angle,robot_3_x,robot_3_y,robot_3_dir_x,robot_3_dir_y,robot_3_angle,robot_4_x,robot_4_y,robot_4_dir_x,robot_4_dir_y,robot_4_angle,robot_5_x,robot_5_y,robot_5_dir_x,robot_5_dir_y,robot_5_angle,robot_6_x,robot_6_y,robot_6_dir_x,robot_6_dir_y,robot_6_angle,robot_7_x,robot_7_y,robot_7_dir_x,robot_7_dir_y,robot_7_angle,robot_8_x,robot_8_y,robot_8_dir_x,robot_8_dir_y,robot_8_angle,robot_9_x,robot_9_y,robot_9_dir_x,robot_9_dir_y,robot_9_angle,robot_10_x,robot_10_y,robot_10_dir_x,robot_10_dir_y,robot_10_angle
1,0.035581350326538086,404.6,247.6,10,442,333,-0.6,-0.8,-2.498091544796509,364,238,0.8,0.6,0.9272952180016122,329,167,-0.9138115486202573,0.40613846605344767,-1.1525719972156676,420,257,0.8,-0.6,2.214297435588181,490,315,-0.8,-0.6,-2.214297435588181,392,178,-0.5144957554275265,0.8574929257125441,-0.5404195002705842,351,336,0.9950371902099892,0.09950371902099892,1.4711276743037347,481,237,0.8944271909999159,-0.4472135954999579,2.0344439357957027,309,293,0.31622776601683794,-0.9486832980505138,2.819842099193151,468,122,0.0,-1.0,3.141592653589793
"""

# Load the data
df = pd.read_csv(io.StringIO(data))

def visualize_robot_swarm():
    """
    Create a comprehensive visualization of the robot swarm data
    """
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Extract robot positions and directions for all frames
    robot_data = {}
    num_robots = 10
    
    for robot_id in range(1, num_robots + 1):
        robot_data[robot_id] = {
            'x': df[f'robot_{robot_id}_x'].values,
            'y': df[f'robot_{robot_id}_y'].values,
            'dir_x': df[f'robot_{robot_id}_dir_x'].values,
            'dir_y': df[f'robot_{robot_id}_dir_y'].values,
            'angle': df[f'robot_{robot_id}_angle'].values
        }
    
    # Color palette for different robots
    colors = plt.cm.tab10(np.linspace(0, 1, num_robots))
    
    # # 1. Trajectory Plot
    # ax1 = plt.subplot(2, 3, 1)
    # for robot_id in range(1, num_robots + 1):
    #     x_pos = robot_data[robot_id]['x']
    #     y_pos = robot_data[robot_id]['y']
    #     ax1.plot(x_pos, y_pos, 'o-', color=colors[robot_id-1], 
    #             label=f'Robot {robot_id}', alpha=0.7, linewidth=2, markersize=4)
    #     # Mark start and end positions
    #     ax1.plot(x_pos[0], y_pos[0], 'o', color=colors[robot_id-1], markersize=8, markeredgecolor='black')
    #     ax1.plot(x_pos[-1], y_pos[-1], 's', color=colors[robot_id-1], markersize=8, markeredgecolor='black')
    
    # # Plot centroid trajectory
    # ax1.plot(df['centroid_x'], df['centroid_y'], 'k*-', linewidth=3, markersize=10, 
    #         label='Centroid', alpha=0.8)
    
    # ax1.set_xlabel('X Position')
    # ax1.set_ylabel('Y Position')
    # ax1.set_title('Robot Trajectories (○ = Start, □ = End)')
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    # ax1.grid(True, alpha=0.3)
    # ax1.set_aspect('equal')
    
    # 2. Final Frame with Direction Vectors
    ax2 = plt.subplot(2,3,2)
    final_frame_idx = -1  # Last frame
    
    for robot_id in range(1, num_robots + 1):
        x = robot_data[robot_id]['x'][final_frame_idx]
        y = -(robot_data[robot_id]['y'][final_frame_idx])
        dx = robot_data[robot_id]['dir_x'][final_frame_idx] * 20  # Scale for visibility
        dy = robot_data[robot_id]['dir_y'][final_frame_idx] * 20
        
        # Plot robot position
        ax2.scatter(x, y, color=colors[robot_id-1], s=100, alpha=0.8, 
                   edgecolor='black', linewidth=1, label=f'Robot {robot_id}')
        # Plot direction vector
        ax2.arrow(x, y, dx, -dy, head_width=5, head_length=8, 
                 fc=colors[robot_id-1], ec='black', alpha=0.7)
        # Add robot ID labels
        ax2.annotate(str(robot_id), (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, fontweight='bold')
    
    # Plot final centroid
    final_centroid_x = df['centroid_x'].iloc[final_frame_idx]
    final_centroid_y = -df['centroid_y'].iloc[final_frame_idx]
    ax2.scatter(final_centroid_x, final_centroid_y, color='red', s=200, 
               marker='*', edgecolor='black', linewidth=2, label='Centroid')
    
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'Final Positions & Directions (Frame {len(df)})')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # # 3. Centroid Movement
    # ax3 = plt.subplot(2, 3, 3)
    # ax3.plot(df['timestamp'], df['centroid_x'], 'b-o', label='Centroid X', linewidth=2)
    # ax3.plot(df['timestamp'], df['centroid_y'], 'r-s', label='Centroid Y', linewidth=2)
    # ax3.set_xlabel('Time (seconds)')
    # ax3.set_ylabel('Position')
    # ax3.set_title('Centroid Position Over Time')
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)
    
    # # 4. Speed Analysis
    # ax4 = plt.subplot(2, 3, 4)
    # speeds = []
    # for robot_id in range(1, num_robots + 1):
    #     robot_speeds = []
    #     x_pos = robot_data[robot_id]['x']
    #     y_pos = robot_data[robot_id]['y']
    #     for i in range(1, len(x_pos)):
    #         dx = x_pos[i] - x_pos[i-1]
    #         dy = y_pos[i] - y_pos[i-1]
    #         speed = np.sqrt(dx**2 + dy**2)
    #         robot_speeds.append(speed)
    #     speeds.append(robot_speeds)
    #     ax4.plot(range(1, len(x_pos)), robot_speeds, 'o-', 
    #             color=colors[robot_id-1], alpha=0.7, label=f'Robot {robot_id}')
    
    # ax4.set_xlabel('Frame Transition')
    # ax4.set_ylabel('Distance Moved')
    # ax4.set_title('Robot Movement Speed Between Frames')
    # ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    # ax4.grid(True, alpha=0.3)
    
    # # 5. Swarm Dispersion
    # ax5 = plt.subplot(2, 3, 5)
    # dispersions = []
    # for frame_idx in range(len(df)):
    #     positions = []
    #     for robot_id in range(1, num_robots + 1):
    #         x = robot_data[robot_id]['x'][frame_idx]
    #         y = robot_data[robot_id]['y'][frame_idx]
    #         positions.append([x, y])
    #     positions = np.array(positions)
    #     centroid = np.mean(positions, axis=0)
    #     distances = np.sqrt(np.sum((positions - centroid)**2, axis=1))
    #     dispersion = np.std(distances)
    #     dispersions.append(dispersion)
    
    # ax5.plot(df['timestamp'], dispersions, 'g-o', linewidth=2, markersize=6)
    # ax5.set_xlabel('Time (seconds)')
    # ax5.set_ylabel('Dispersion (std of distances from centroid)')
    # ax5.set_title('Swarm Cohesion Over Time')
    # ax5.grid(True, alpha=0.3)
    
    # # 6. Direction Distribution
    # ax6 = plt.subplot(2, 3, 6)
    # all_angles = []
    # for robot_id in range(1, num_robots + 1):
    #     all_angles.extend(robot_data[robot_id]['angle'])
    
    # ax6.hist(all_angles, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    # ax6.set_xlabel('Angle (radians)')
    # ax6.set_ylabel('Frequency')
    # ax6.set_title('Distribution of Robot Orientations')
    # ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_animation():
    """
    Create an animated visualization of the robot swarm
    """
    print("Creating animation...")
    
    # Extract robot data
    robot_data = {}
    num_robots = 10
    
    for robot_id in range(1, num_robots + 1):
        robot_data[robot_id] = {
            'x': df[f'robot_{robot_id}_x'].values,
            'y': df[f'robot_{robot_id}_y'].values,
            'dir_x': df[f'robot_{robot_id}_dir_x'].values,
            'dir_y': df[f'robot_{robot_id}_dir_y'].values,
        }
    
    # Set up animation
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_robots))
    
    # Find plot bounds
    all_x = []
    all_y = []
    for robot_id in range(1, num_robots + 1):
        all_x.extend(robot_data[robot_id]['x'])
        all_y.extend(robot_data[robot_id]['y'])
    
    margin = 30
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Initialize plot elements
    robot_dots = []
    robot_arrows = []
    robot_trails = []
    
    for robot_id in range(1, num_robots + 1):
        # Robot position dot
        dot, = ax.plot([], [], 'o', color=colors[robot_id-1], markersize=10, 
                      markeredgecolor='black', linewidth=1)
        robot_dots.append(dot)
        
        # Direction arrow (will be updated each frame)
        arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=colors[robot_id-1], lw=2))
        robot_arrows.append(arrow)
        
        # Trail
        trail, = ax.plot([], [], '-', color=colors[robot_id-1], alpha=0.5, linewidth=1)
        robot_trails.append(trail)
    
    # Centroid
    centroid_dot, = ax.plot([], [], '*', color='red', markersize=15, 
                           markeredgecolor='black', linewidth=2)
    centroid_trail, = ax.plot([], [], 'k-', linewidth=2, alpha=0.7)
    
    # Frame counter
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Robot Swarm Animation')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    def animate(frame_idx):
        # Update robot positions and directions
        for robot_id in range(1, num_robots + 1):
            idx = robot_id - 1
            x = robot_data[robot_id]['x'][frame_idx]
            y = robot_data[robot_id]['y'][frame_idx]
            dx = robot_data[robot_id]['dir_x'][frame_idx] * 25
            dy = robot_data[robot_id]['dir_y'][frame_idx] * 25
            
            # Update position
            robot_dots[idx].set_data([x], [y])
            
            # Update direction arrow
            robot_arrows[idx].xy = (x + dx, y + dy)
            robot_arrows[idx].xytext = (x, y)
            
            # Update trail
            trail_x = robot_data[robot_id]['x'][:frame_idx+1]
            trail_y = robot_data[robot_id]['y'][:frame_idx+1]
            robot_trails[idx].set_data(trail_x, trail_y)
        
        # Update centroid
        centroid_x = df['centroid_x'].iloc[frame_idx]
        centroid_y = df['centroid_y'].iloc[frame_idx]
        centroid_dot.set_data([centroid_x], [centroid_y])
        
        # Update centroid trail
        centroid_trail_x = df['centroid_x'][:frame_idx+1]
        centroid_trail_y = df['centroid_y'][:frame_idx+1]
        centroid_trail.set_data(centroid_trail_x, centroid_trail_y)
        
        # Update frame counter
        frame_text.set_text(f'Frame: {frame_idx+1}/{len(df)}\nTime: {df["timestamp"].iloc[frame_idx]:.3f}s')
        
        return robot_dots + robot_arrows + robot_trails + [centroid_dot, centroid_trail, frame_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(df), interval=500, blit=False, repeat=True)
    plt.show()
    
    return anim

# Run the visualizations
print("Robot Swarm Data Analysis")
print("=" * 40)
print(f"Total frames: {len(df)}")
print(f"Time span: {df['timestamp'].iloc[0]:.3f}s to {df['timestamp'].iloc[-1]:.3f}s")
print(f"Number of active robots: {df['active_robots'].iloc[0]}")
print()

# Create static visualization
visualize_robot_swarm()

# Create animation
animation = create_animation()

print("\nVisualization complete!")
print("\nData insights:")
print(f"- Centroid moves from ({df['centroid_x'].iloc[0]:.1f}, {df['centroid_y'].iloc[0]:.1f}) to ({df['centroid_x'].iloc[-1]:.1f}, {df['centroid_y'].iloc[-1]:.1f})")
print(f"- Average centroid position: ({df['centroid_x'].mean():.1f}, {df['centroid_y'].mean():.1f})")
print("- Each robot has unique movement patterns and orientations")
print("- The swarm appears to be performing coordinated movement behaviors")