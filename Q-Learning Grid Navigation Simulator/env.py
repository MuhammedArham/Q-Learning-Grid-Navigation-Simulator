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

    def step(self, action):
        """
        Take an action in the environment, resulting in a state transition.

        Args:
            action (int): The action taken by the agent (0: up, 1: down, 2: left, 3: right).

        Returns:
            np.array: The new state (position) of the agent.
            int: The reward for the action taken.
            bool: Whether the goal has been reached (episode termination).
            dict: Extra information (unused here).
        """
        x, y = self.state  # Get the agent's current position
        next_x, next_y = x, y  # Initialize the next state as the current state

        # Update position based on the action
        if action == 0 and x > 0:  # Move up
            next_x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Move down
            next_x += 1
        elif action == 2 and y > 0:  # Move left
            next_y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Move right
            next_y += 1

        # Check if the next position is an obstacle
        if (next_x, next_y) in self.obstacles:
            # If the next position is an obstacle, stay in the current position
            next_x, next_y = x, y
            reward = -5  # Penalty for hitting an obstacle
        else:
            reward = -1  # Normal step penalty

        self.state = (next_x, next_y)  # Update the agent's position

        # Check if the agent has reached the goal
        if self.state == self.goal:
            return (
                self.state,
                100,
                True,
                {},
            )  # Reward of 100 for reaching the goal

        return self.state, reward, False, {}  # Continue the episode

    def render(
        self,
        delay=0.1,
        mode="human",
        episode=1,
        learning_type="Q-learning",
        availability=None,
        accuracy=None,
    ):
        """
        Render the grid environment, displaying the agent, goal, and obstacles.
        Also display information such as episode number, learning type, and optionally availability and accuracy.

        Args:
            delay (float): Delay between frames (to control speed of rendering).
            mode (str): Rendering mode (unused in this implementation).
            episode (int): Current episode number.
            learning_type (str): The type of learning algorithm being used (e.g., Q-learning, SARSA).
            availability (float): Teacher availability (optional, as a percentage).
            accuracy (float): Teacher accuracy (optional, as a percentage).
        """
        if not self.render_initialized:
            # Initialize Pygame only when rendering for the first time
