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
