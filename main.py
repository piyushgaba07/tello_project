#!/usr/bin/env python3
"""
Tello Drone Control with GUI

This application provides a graphical interface for controlling the Tello drone
using different input methods: gestures, voice commands, and vision analysis.

Author: Piyush Gaba
"""

import sys
import os
import logging
import builtins
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Store original print function
_original_print = builtins.print

# DO NOT replace print function yet - first import the modules

# Import the main app
try:
    from gui.drone_app import main
except Exception as e:
    # If error occurs during import, show error using original print
    _original_print(f"Error importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

# NOW import logger and replace print
try:
    from utils.logger import custom_print
    # Replace the print function AFTER imports to avoid circular imports
    builtins.print = custom_print
except Exception as e:
    _original_print(f"Error setting up logger: {e}")
    traceback.print_exc()

if __name__ == "__main__":
    try:
        # Run the main application
        main()
    except KeyboardInterrupt:
        # Restore original print before printing
        builtins.print = _original_print
        print("\nApplication terminated by user.")
    except Exception as e:
        # Restore original print before printing error
        builtins.print = _original_print
        print(f"Uncaught exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original print before exiting
        builtins.print = _original_print 