import numpy as np
import matplotlib.pyplot as plt
from src.lidar_visibility_map import LidarVisibilityMap, LidarVisibilityParams

def main():
    # Create parameters for the visibility map
    params = LidarVisibilityParams(
        x_size=50,
        y_size=50,
        resolution=0.5,
        max_range=20.0,
        visibility_threshold=0.3,
        decay_rate=0.05
    )
    
    # Create the visibility map
    visibility_map = LidarVisibilityMap(params)
    
    # Simulate agent at position (25, 25) with lidar readings
    agent_pos = np.array([25.0, 25.0])
    
    # Simulate lidar readings (360 degrees, 1-degree resolution)
    num_angles = 360
    lidar_angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # Simulate some obstacles
    obstacles = [
        (35, 25),  # obstacle to the right
        (25, 35),  # obstacle above
        (15, 25),  # obstacle to the left
        (25, 15),  # obstacle below
    ]
    
    # Generate lidar readings
    lidar_data = np.ones(num_angles) * params.max_range
    for obstacle in obstacles:
        for i, angle in enumerate(lidar_angles):
            # Calculate direction vector
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Calculate vector from agent to obstacle
            ox = obstacle[0] - agent_pos[0]
            oy = obstacle[1] - agent_pos[1]
            
            # Calculate distance to obstacle
            distance = np.sqrt(ox**2 + oy**2)
            
            # Calculate angle to obstacle
            obstacle_angle = np.arctan2(oy, ox)
            
            # Normalize angle difference to [-pi, pi]
            angle_diff = np.abs(angle - obstacle_angle)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            # If the angle difference is small and the obstacle is closer than the current reading
            if angle_diff < 0.1 and distance < lidar_data[i]:
                lidar_data[i] = distance
    
    # Update visibility map
    visibility_map.update_from_lidar(agent_pos, lidar_data, lidar_angles)
    
    # Plot the maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot visibility map
    axes[0].imshow(visibility_map.visibility_map, vmin=0.0, vmax=1.0, 
                  extent=[0, params.x_size, 0, params.y_size], 
                  origin='lower', cmap='viridis')
    axes[0].set_title('Visibility Map')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Plot occupancy map
    axes[1].imshow(visibility_map.occupancy_map, vmin=0.0, vmax=1.0, 
                  extent=[0, params.x_size, 0, params.y_size], 
                  origin='lower', cmap='binary')
    axes[1].set_title('Occupancy Map')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Plot failure map
    failure_map = visibility_map.get_failure_map()
    axes[2].imshow(failure_map, vmin=0.0, vmax=1.0, 
                  extent=[0, params.x_size, 0, params.y_size], 
                  origin='lower', cmap='Reds')
    axes[2].set_title('Failure Map')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    # Plot the agent and lidar rays
    for ax in axes:
        # Plot agent
        ax.plot(agent_pos[0], agent_pos[1], 'ro', markersize=5)
        
        # Plot lidar rays
        for i, (distance, angle) in enumerate(zip(lidar_data, lidar_angles)):
            if i % 10 == 0:  # Plot every 10th ray for clarity
                end_x = agent_pos[0] + distance * np.cos(angle)
                end_y = agent_pos[1] + distance * np.sin(angle)
                ax.plot([agent_pos[0], end_x], [agent_pos[1], end_y], 'y-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('misc/lidar_visibility_test.png')
    plt.show()

if __name__ == "__main__":
    main()