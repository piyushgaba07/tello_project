"""
Utility modules for the Tello Drone Control application.
"""

from utils.config import *
from utils.logger import logger, custom_print

# Import builtins, but DON'T replace print yet to avoid recursion
# The replacement will be done in the main.py when needed
import builtins
# builtins.print = custom_print  # <-- Commented out to avoid recursion error 