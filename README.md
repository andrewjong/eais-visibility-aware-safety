# Visibility-Aware Safety for Autonomous Racing

This repository contains code for implementing visibility-aware safety for autonomous racing using Hamilton-Jacobi reachability analysis. The code is adapted from the original EAIS (Environment-Aware Interactive Safety) repository to focus on car racing with visibility-aware safety using the racecar_gym environment.

## Overview

The main idea is to use lidar observations from a racing car to build a visibility map, which is then used to identify potentially unsafe regions (areas that are not visible to the agent). Hamilton-Jacobi reachability analysis is used to compute a safety controller that keeps the agent away from these unsafe regions.

## Key Components

1. **Lidar Visibility Map**: Converts lidar observations into a visibility map that identifies areas that are visible to the agent.
2. **Racecar Environment**: Wrapper for the racecar_gym environment that integrates with the visibility-aware safety approach.
3. **Hamilton-Jacobi Reachability**: Computes a safety controller that keeps the agent away from unsafe regions.
4. **MPPI Controller**: A Model Predictive Path Integral controller that serves as the nominal controller.

## Installation Instructions

Install the conda environment using the provided `environment.yml` file. This will set up all the necessary dependencies for the project.
```bash
conda env create -f environment.yml
```

Then install the local racecar_gym package using pip:
```bash
cd racecar_gym/
pip install -e .
```

## Files

- `src/lidar_visibility_map.py`: Implements the visibility map based on lidar data.
- `envs/racecar_env.py`: Wrapper for the racecar_gym environment.
- `run/racecar_visibility_safety.py`: Main script for running the visibility-aware safety approach.

## Usage

To run the visibility-aware safety approach:

```bash
python run/racecar_visibility_safety.py
```

## Adaptation from Original Code

The original code used a smoke environment with a Gaussian Process to model smoke density. This adaptation replaces the smoke environment with a racecar environment and uses lidar observations to build a visibility map. The key changes are:

1. Replaced the smoke environment with the racecar_gym environment.
2. Replaced the Gaussian Process smoke density model with a lidar-based visibility map.
3. Adapted the Hamilton-Jacobi reachability analysis to work with the visibility map.
4. Updated the visualization to show the racecar environment, visibility map, and safety controller.

## References

- [Hamilton-Jacobi Reachability: A Brief Overview and Recent Advances](https://arxiv.org/abs/1709.07523)
- [Environment-Aware Interactive Safety: Leveraging Prediction and Uncertainty for Human-Robot Interaction](https://arxiv.org/abs/2203.02432)
- [racecar_gym: A Modular Reinforcement Learning Framework for Autonomous Racing](https://github.com/axelbr/racecar_gym)