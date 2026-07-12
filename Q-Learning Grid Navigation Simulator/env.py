import os
import time

import numpy as np


# Class representing a static grid environment
class StaticGridEnv:
    def __init__(self, seed=None):
        """
        Initialize the static grid environment.

        Args:
            seed (int): Optional random seed for reproducibility.
        """
        self.grid_size = 10  # Size of the grid (10x10)
        self.cell_size = 64  # Pixel size of each cell in the grid (64x64 pixels)

        # Define obstacles as a list of coordinates (x, y)
        self.obstacles = [(3,0),(2,1),(1,2),(8,7),(7,8),(6,9),
                          (8,1),(7,1),(6,1),(5,2),(4,3),(3,4),(2,5),(1,5),(0,5),
                          (9,4),(8,4),(7,4),(6,5),(5,6),(4,7),(3,8),(2,8),(1,8)]

        # Define the goal position at the bottom-right corner of the grid
        self.goal = (self.grid_size - 1, self.grid_size - 1)

        self.state = None  # The agent's current position will be set later

        # Set random seed if provided to ensure consistent results
        if seed is not None:
            np.random.seed(seed)

        # Define the action space: 4 possible directions (up, down, left, right)
        self.action_space = 4

        # Observation space is the grid itself, represented by its dimensions
        self.observation_space = (self.grid_size, self.grid_size)

        # Pygame screen initialization is delayed until rendering to avoid immediate window creation
        self.screen = None
        self.render_initialized = False  # Flag to indicate if rendering has been set up

    def reset(self):
        """
        Reset the environment by selecting a random start position for the agent,
        avoiding obstacles and the goal.

        Returns:
            np.array: The initial position of the agent.
        """
        while True:
            # Randomly select starting coordinates (x, y) in the grid
            start_x = np.random.randint(0, self.grid_size)
            start_y = np.random.randint(0, self.grid_size)

            # Ensure the starting position is not on an obstacle or the goal
            if (start_x, start_y) not in self.obstacles and (
                start_x,
                start_y,
            ) != self.goal:
                self.state = (start_x, start_y)
                break

        return self.state

