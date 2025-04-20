import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Arrow
from src.lidar_visibility_map import LidarVisibilityMap, LidarVisibilityParams

@dataclass
class RacecarEnvParams:
    """Parameters for the racecar environment."""
    world_x_size: int = field(default=50)
    world_y_size: int = field(default=50)
    max_steps: int = field(default=1000)
    render: bool = field(default=False)
    clock: float = field(default=0.1)
    goal_location: tuple[float, float] | None = field(default=None)
    track_name: str = field(default="montreal")  # Name of the track to use

@dataclass
class RacecarParams:
    """Parameters for the racecar."""
    v_min: float = field(default=0.0)
    v_max: float = field(default=5.0)
    omega_min: float = field(default=-4.0)
    omega_max: float = field(default=4.0)
    world_x_size: int = field(default=50)
    world_y_size: int = field(default=50)

class RacecarEnvWrapper(gym.Wrapper):
    """
    Wrapper for the racecar_gym environment that integrates with the visibility-aware safety approach.
    
    This wrapper adapts the racecar_gym environment to work with the visibility-aware safety approach
    by converting lidar observations to a visibility map.
    """
    def __init__(self, env_params: RacecarEnvParams, racecar_params: RacecarParams):
        """
        Initialize the racecar environment wrapper.
        
        Args:
            env_params: Parameters for the environment
            racecar_params: Parameters for the racecar
        """
        import gymnasium
        import racecar_gym.envs.gym_api
        
        # Create the racecar environment
        env_id = f"SingleAgent{env_params.track_name.capitalize()}-v0"
        render_mode = 'human' if env_params.render else None
        env = gymnasium.make(env_id, render_mode=render_mode)
        
        super().__init__(env)
        
        self.env_params = env_params
        self.racecar_params = racecar_params
        
        # Override the action space to match our expected format
        self.action_space = spaces.Box(
            low=np.array([self.racecar_params.v_min, self.racecar_params.omega_min]),
            high=np.array([self.racecar_params.v_max, self.racecar_params.omega_max]),
            shape=(2,)
        )
        
        # Create a visibility map
        visibility_params = LidarVisibilityParams(
            x_size=self.env_params.world_x_size,
            y_size=self.env_params.world_y_size,
            resolution=0.5,  # 0.5 meters per cell
            max_range=20.0,  # Maximum lidar range
            visibility_threshold=0.3,  # Threshold for considering a cell visible
            decay_rate=0.05  # Rate at which visibility decays with distance
        )
        self.visibility_map = LidarVisibilityMap(visibility_params)
        
        # Initialize visualization
        self.window = {"fig": None, "ax": None}
        self.clock = self.env_params.clock
        
        # Track trajectory
        self.trajectory = []
        
    def reset(self, initial_state=None, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            initial_state: Initial state (optional)
            seed: Random seed (optional)
            options: Reset options (optional)
            
        Returns:
            Observation and info
        """
        self.window = {"fig": None, "ax": None}
        self.trajectory = []
        
        # Reset the underlying environment
        if options is None:
            options = {"mode": "grid"}
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Convert observation to our format
        state = self._convert_observation(obs)
        
        # Update the visibility map
        self._update_visibility_map(obs)
        
        return state, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take [velocity, steering]
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Convert action to racecar_gym format if needed
        # In racecar_gym, action is [steering, acceleration]
        racecar_action = np.array([action[1], action[0]])
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(racecar_action)
        
        # Convert observation to our format
        state = self._convert_observation(obs)
        
        # Update the visibility map
        self._update_visibility_map(obs)
        
        # Track trajectory
        self.trajectory.append(state[:2])
        
        # Render if needed
        if self.env_params.render:
            self._render_frame()
        
        return state, reward, terminated, truncated, info
    
    def _convert_observation(self, obs):
        """
        Convert racecar_gym observation to our format.
        
        Args:
            obs: Observation from racecar_gym
            
        Returns:
            State vector [x, y, theta, visibility]
        """
        # Extract position and orientation
        position = obs['pose'][0:2]  # [x, y]
        orientation = obs['pose'][2]  # theta
        
        # Get visibility at the current position
        visibility = 1.0 - self.visibility_map.get_visibility_at(position[0], position[1])
        
        # Create state vector [x, y, theta, visibility]
        state = np.array([position[0], position[1], orientation, visibility])
        
        return state
    
    def _update_visibility_map(self, obs):
        """
        Update the visibility map based on lidar observations.
        
        Args:
            obs: Observation from racecar_gym
        """
        # Extract lidar data
        lidar_data = obs['lidar']
        
        # Extract position and orientation
        position = obs['pose'][0:2]  # [x, y]
        orientation = obs['pose'][2]  # theta
        
        # Generate lidar angles (assuming 360 lidar readings over 2π radians)
        num_angles = len(lidar_data)
        lidar_angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        # Adjust angles based on agent orientation
        lidar_angles = (lidar_angles + orientation) % (2*np.pi)
        
        # Update the visibility map
        self.visibility_map.update_from_lidar(position, lidar_data, lidar_angles)
    
    def _render_frame(self, fig=None, ax=None):
        """
        Render the current state of the environment.
        
        Args:
            fig: Figure to render on (optional)
            ax: Axis to render on (optional)
        """
        if self.window["fig"] is None:
            if fig is not None and ax is not None:
                self.window["fig"] = fig
                self.window["ax"] = ax
            else:
                self.window["fig"], self.window["ax"] = plt.subplots(figsize=(8, 8))
        
        # Clear the axis
        self.window["ax"].clear()
        
        # Plot the failure map
        failure_map = self.visibility_map.get_failure_map()
        self.window["ax"].imshow(
            failure_map,
            extent=[0, self.env_params.world_x_size, 0, self.env_params.world_y_size],
            origin='lower',
            cmap='Reds',
            alpha=0.7
        )
        
        # Get the current state
        state = self._convert_observation(self.env.unwrapped._observations)
        
        # Plot the agent
        self.window["ax"].arrow(
            state[0], state[1],
            np.cos(state[2])*1.0, np.sin(state[2])*1.0,
            head_width=0.5, head_length=0.5, fc='b', ec='b'
        )
        
        # Plot the goal if it exists
        if self.env_params.goal_location is not None:
            circle = Circle(
                (self.env_params.goal_location[0], self.env_params.goal_location[1]),
                radius=1.0, color='g', fill=True, alpha=0.8
            )
            self.window["ax"].add_patch(circle)
        
        # Plot the trajectory
        if len(self.trajectory) > 0:
            trajectory = np.array(self.trajectory)
            self.window["ax"].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1, alpha=0.5)
        
        # Set axis limits
        self.window["ax"].set_xlim(0, self.env_params.world_x_size)
        self.window["ax"].set_ylim(0, self.env_params.world_y_size)
        
        # Set title
        self.window["ax"].set_title("Racecar Environment with Visibility-Aware Safety")
        
        # Redraw the plot
        self.window["fig"].canvas.draw()
        self.window["fig"].canvas.flush_events()
        plt.pause(self.clock)
    
    def close(self):
        """Close the environment."""
        self.window = {"fig": None, "ax": None}
        return self.env.close()


if __name__ == "__main__":
    # Test the racecar environment wrapper
    env_params = RacecarEnvParams(
        world_x_size=50,
        world_y_size=50,
        max_steps=1000,
        render=True,
        clock=0.1,
        goal_location=(40, 40),
        track_name="montreal"
    )
    
    racecar_params = RacecarParams()
    
    env = RacecarEnvWrapper(env_params, racecar_params)
    state, _ = env.reset()
    
    print("Initial state:", state)
    
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()