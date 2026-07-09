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
