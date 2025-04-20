import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

@dataclass
class LidarVisibilityParams:
    """Parameters for the lidar-based visibility map."""
    x_size: int  # Size of the map in x direction
    y_size: int  # Size of the map in y direction
    resolution: float  # Resolution of the map (meters per cell)
    max_range: float  # Maximum range of the lidar (meters)
    visibility_threshold: float  # Threshold for considering a cell visible (0-1)
    decay_rate: float = 0.1  # Rate at which visibility decays with distance


class LidarVisibilityMap:
    """
    Creates a visibility map from lidar data.
    
    The visibility map represents areas that are visible to the agent (1) and areas that are not (0).
    This replaces the smoke density model in the original code.
    """
    def __init__(self, params: LidarVisibilityParams):
        """
        Initialize the visibility map.
        
        Args:
            params: Parameters for the visibility map
        """
        self.params = params
        
        # Initialize the visibility map (0 = not visible, 1 = visible)
        self.visibility_map = np.zeros((
            int(self.params.y_size / self.params.resolution),
            int(self.params.x_size / self.params.resolution)
        ))
        
        # Initialize the occupancy map (0 = free, 1 = occupied)
        self.occupancy_map = np.zeros_like(self.visibility_map)
        
        # Keep track of all cells that have been observed
        self.observed_map = np.zeros_like(self.visibility_map)
        
    def update_from_lidar(self, 
                         position: np.ndarray, 
                         lidar_data: np.ndarray, 
                         lidar_angles: np.ndarray) -> None:
        """
        Update the visibility map based on lidar data.
        
        Args:
            position: Agent position [x, y]
            lidar_data: Array of lidar distances
            lidar_angles: Array of lidar angles (relative to agent orientation)
            orientation: Agent orientation (radians)
        """
        # Reset visibility map
        self.visibility_map = np.zeros_like(self.visibility_map)
        
        # Convert agent position to grid coordinates
        agent_x, agent_y = self._world_to_grid(position[0], position[1])
        
        # Mark the agent's position as visible
        if 0 <= agent_x < self.visibility_map.shape[1] and 0 <= agent_y < self.visibility_map.shape[0]:
            self.visibility_map[agent_y, agent_x] = 1
            self.observed_map[agent_y, agent_x] = 1
        
        # Process each lidar ray
        for i, (distance, angle) in enumerate(zip(lidar_data, lidar_angles)):
            # Skip invalid readings
            if distance <= 0:
                continue
                
            # Calculate endpoint of the ray in world coordinates
            end_x = position[0] + distance * np.cos(angle)
            end_y = position[1] + distance * np.sin(angle)
            
            # Convert to grid coordinates
            end_grid_x, end_grid_y = self._world_to_grid(end_x, end_y)
            
            # Use Bresenham's line algorithm to find cells along the ray
            cells = self._bresenham_line(agent_x, agent_y, end_grid_x, end_grid_y)
            
            # Mark cells along the ray as visible
            for j, (cell_x, cell_y) in enumerate(cells):
                if 0 <= cell_x < self.visibility_map.shape[1] and 0 <= cell_y < self.visibility_map.shape[0]:
                    # Calculate distance from agent to this cell
                    cell_distance = j * self.params.resolution
                    
                    # Visibility decays with distance
                    visibility = max(0, 1 - self.params.decay_rate * cell_distance)
                    
                    # Mark cell as visible
                    self.visibility_map[cell_y, cell_x] = max(self.visibility_map[cell_y, cell_x], visibility)
                    self.observed_map[cell_y, cell_x] = 1
                    
                    # If this is the endpoint, mark it as occupied
                    if j == len(cells) - 1:
                        self.occupancy_map[cell_y, cell_x] = 1
    
    def get_visibility_at(self, x: float, y: float) -> float:
        """
        Get the visibility value at a specific world position.
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            
        Returns:
            Visibility value (0-1)
        """
        grid_x, grid_y = self._world_to_grid(x, y)
        
        if 0 <= grid_x < self.visibility_map.shape[1] and 0 <= grid_y < self.visibility_map.shape[0]:
            return self.visibility_map[grid_y, grid_x]
        else:
            return 0.0
    
    def get_failure_map(self) -> np.ndarray:
        """
        Get the failure map based on the visibility map.
        
        The failure map is the inverse of the visibility map, where 1 represents
        areas that are not visible (and thus potentially unsafe).
        
        Returns:
            Failure map as a numpy array
        """
        # Areas that have been observed but are not currently visible are considered safe
        # Areas that have never been observed are considered unsafe
        failure_map = 1 - (self.visibility_map + (self.observed_map * (1 - self.visibility_map)))
        
        # Apply threshold to determine failure regions
        failure_map = (failure_map > (1 - self.params.visibility_threshold)).astype(float)
        
        return failure_map
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            
        Returns:
            Tuple of (grid_x, grid_y)
        """
        grid_x = int(x / self.params.resolution)
        grid_y = int(y / self.params.resolution)
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            grid_x: X coordinate in grid
            grid_y: Y coordinate in grid
            
        Returns:
            Tuple of (world_x, world_y)
        """
        world_x = (grid_x + 0.5) * self.params.resolution
        world_y = (grid_y + 0.5) * self.params.resolution
        return world_x, world_y
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for finding cells along a line.
        
        Args:
            x0, y0: Starting point
            x1, y1: Ending point
            
        Returns:
            List of (x, y) tuples representing cells along the line
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        return cells
    
    def plot_maps(self, fig: Optional[plt.Figure] = None, axes: Optional[List[plt.Axes]] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot the visibility and occupancy maps.
        
        Args:
            fig: Figure to plot on (optional)
            axes: List of axes to plot on (optional)
            
        Returns:
            Tuple of (figure, axes)
        """
        if fig is None or axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot visibility map
        if axes[0].images:
            axes[0].images[0].set_array(self.visibility_map)
        else:
            axes[0].imshow(self.visibility_map, vmin=0.0, vmax=1.0, 
                          extent=[0, self.params.x_size, 0, self.params.y_size], 
                          origin='lower', cmap='viridis')
            axes[0].set_title('Visibility Map')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
        
        # Plot occupancy map
        if axes[1].images:
            axes[1].images[0].set_array(self.occupancy_map)
        else:
            axes[1].imshow(self.occupancy_map, vmin=0.0, vmax=1.0, 
                          extent=[0, self.params.x_size, 0, self.params.y_size], 
                          origin='lower', cmap='binary')
            axes[1].set_title('Occupancy Map')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
        
        # Plot failure map
        failure_map = self.get_failure_map()
        if axes[2].images:
            axes[2].images[0].set_array(failure_map)
        else:
            axes[2].imshow(failure_map, vmin=0.0, vmax=1.0, 
                          extent=[0, self.params.x_size, 0, self.params.y_size], 
                          origin='lower', cmap='Reds')
            axes[2].set_title('Failure Map')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
        
        fig.tight_layout()
        return fig, axes


if __name__ == "__main__":
    # Test the visibility map with simulated lidar data
    params = LidarVisibilityParams(
        x_size=50,
        y_size=50,
        resolution=0.5,
        max_range=20.0,
        visibility_threshold=0.3,
        decay_rate=0.05
    )
    
    visibility_map = LidarVisibilityMap(params)
    
    # Simulate agent at position (25, 25) with lidar readings
    agent_pos = np.array([25.0, 25.0])
    agent_orientation = 0.0  # facing right
    
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
    fig, axes = visibility_map.plot_maps()
    
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
    
    plt.show()