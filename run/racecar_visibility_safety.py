import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Arrow
from src.mppi import Navigator, dubins_dynamics_tensor
from reachability.warm_start_solver import WarmStartSolver, WarmStartSolverConfig
from envs.racecar_env import RacecarEnvWrapper, RacecarEnvParams, RacecarParams

def main():
    # Create environment parameters
    env_params = RacecarEnvParams()
    env_params.world_x_size = 50
    env_params.world_y_size = 50
    env_params.max_steps = 2000
    env_params.render = False
    env_params.goal_location = (40, 40)
    env_params.track_name = "montreal"  # Use the Montreal track
    
    # Create racecar parameters
    racecar_params = RacecarParams()
    racecar_params.world_x_size = env_params.world_x_size
    racecar_params.world_y_size = env_params.world_y_size
    
    # Create the environment
    env = RacecarEnvWrapper(env_params, racecar_params)
    
    # Reset the environment
    state, _ = env.reset()
    
    # Create the Hamilton-Jacobi reachability solver
    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=[50, 50, 40],  # [x_cells, y_cells, theta_cells]
            domain=[[0, 0, 0], [env_params.world_x_size, env_params.world_y_size, 2*np.pi]],
            mode="brt",
            accuracy="medium",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )
    
    # Create a nominal controller (MPPI)
    nom_controller = Navigator()
    nom_controller.set_odom(state[:2], state[2])
    nom_controller.set_map(env.visibility_map.get_failure_map(), 
                          [env_params.world_x_size, env_params.world_y_size], 
                          [0, 0], 
                          1.0)
    nom_controller.set_goal(list(env_params.goal_location))
    
    # Set up nominal action
    NOMINAL_ACTION_V = 2.0
    nominal_action_w = nom_controller.get_command().item()
    nominal_action = np.array([NOMINAL_ACTION_V, nominal_action_w])
    
    # Set up visualization
    f = plt.figure(figsize=(15, 5))
    gs = f.add_gridspec(1, 3)
    ax_env = f.add_subplot(gs[0])
    ax_visibility = f.add_subplot(gs[1])
    ax_fail = f.add_subplot(gs[2])
    
    # Track trajectories
    traj_safe = []
    traj_unsafe = []
    
    # Set up update interval for the visibility map and solver
    update_interval = 5
    values = None
    
    # Main loop
    for t in range(1, env_params.max_steps):
        # Update the visibility map and solver periodically
        if t % update_interval == 0:
            # Get the failure map from the visibility map
            failure_map = env.visibility_map.get_failure_map()
            
            # Update the nominal controller with the new failure map
            nom_controller.set_map(failure_map, 
                                  [env_params.world_x_size, env_params.world_y_size], 
                                  [0, 0], 
                                  1.0)
            
            # Solve the Hamilton-Jacobi reachability problem
            if np.all(failure_map == 1):
                # If the entire map is unsafe, don't compute a value function
                values = None
            else:
                # Otherwise, compute the value function
                values = solver.solve(failure_map.T, target_time=-10.0, dt=0.1, epsilon=0.0001)
        
        # Get the nominal action from the controller
        nominal_action = nom_controller.get_command()
        nominal_action = np.array([NOMINAL_ACTION_V, nominal_action.item()])
        
        # Compute the safe action
        if values is not None:
            # Use the HJ reachability solver to compute a safe action
            safe_action, _, _ = solver.compute_safe_control(
                state[0:3],  # [x, y, theta]
                nominal_action,  # [v, omega]
                action_bounds=np.array([[0.0, 5.0], [-4.0, 4.0]]),  # [v_bounds, omega_bounds]
                values=values  # Value function
            )
        else:
            # If no value function is available, use the nominal action
            safe_action = nominal_action
        
        # Take a step in the environment
        state, reward, terminated, truncated, info = env.step(safe_action)
        
        # Track trajectory based on safety
        if state[3] > 0.7:  # If visibility is low (unsafe)
            traj_unsafe.append(state[0:2])
        else:  # If visibility is high (safe)
            traj_safe.append(state[0:2])
        
        # Check if the episode is done
        if terminated or truncated:
            break
        
        # Update the nominal controller with the new state
        nom_controller.set_odom(state[:2], state[2])
        
        # Render the environment
        env._render_frame(fig=f, ax=ax_env)
        
        # Plot the visibility map
        env.visibility_map.plot_maps(fig=f, axes=[ax_visibility, ax_visibility, ax_fail])
        
        # Clear arrows from the failure map
        for arrow in ax_fail.patches:
            if isinstance(arrow, (FancyArrow, Arrow)):
                arrow.remove()
        
        # Clear collections from the failure map
        for coll in ax_fail.collections:
            coll.remove()
        
        # Determine if the current state is safe
        if values is not None:
            is_safe, _, _ = solver.check_if_safe(state[:3], values)
            color_robot = 'g' if is_safe else 'r'
        else:
            color_robot = 'g'
        
        # Plot the agent on the failure map
        ax_fail.arrow(
            state[0], state[1],
            np.cos(state[2])*1.0, np.sin(state[2])*1.0,
            head_width=0.5, head_length=0.5, fc=color_robot, ec=color_robot
        )
        
        # Plot trajectories
        if len(traj_safe) > 0:
            ax_fail.scatter(
                np.array(traj_safe)[:, 0], np.array(traj_safe)[:, 1],
                color='green', label='Safe', marker='.', s=0.2
            )
        if len(traj_unsafe) > 0:
            ax_fail.scatter(
                np.array(traj_unsafe)[:, 0], np.array(traj_unsafe)[:, 1],
                color='red', label='Unsafe', marker='.', s=0.2
            )
        
        # Plot the value function contour if available
        if values is not None:
            # Get the value function slice at the current orientation
            state_ind = solver._state_to_grid(state[:3])
            z = values[:, :, state_ind[2]].T
            z_mask = z > 0.1
            
            # Plot the contour of the safe set
            ax_fail.contour(
                np.arange(env_params.world_x_size),
                np.arange(env_params.world_y_size),
                z_mask, levels=[0.5], colors='green'
            )
        
        # Update the plot
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    # Save the final figure
    figure_name = input('Enter the name of the figure: ')
    f.savefig(f'misc/{figure_name}.png', bbox_inches='tight')
    
    # Print statistics
    print(f'Terminated at time {t*env_params.clock} seconds')
    print(f'Time in unsafe regions: {len(traj_unsafe) * env_params.clock} seconds')
    print(f'Time in safe regions: {len(traj_safe) * env_params.clock} seconds')
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()