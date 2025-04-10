"""
Configuration settings for the Tello Drone Control application.
"""

# Style constants
BUTTON_BG = "#f0f0f0"
HEADER_BG = "#e1e1e1"
FRAME_BG = "#f5f5f5"
ACCENT_COLOR = "#3498db"

# Log levels with colors
LOG_LEVELS = {
    "INFO": "#0066cc",     # Blue
    "ERROR": "#cc0000",    # Red
    "COMMAND": "#009933",  # Green
    "GESTURE": "#9933cc",  # Purple
    "SPEECH": "#cc6600",   # Orange
    "VLM": "#663300",      # Brown
    "DEBUG": "#666666"     # Gray
}

# Control modes
MODES = ["gesture", "audio", "vlm", "idle"]
DEFAULT_MODE = "idle"

# VLM settings
VLM_COOLDOWN = 3  # seconds between VLM queries

# Queue settings
COMMAND_QUEUE_SIZE = 5
FRAME_QUEUE_SIZE = 1
LOG_QUEUE_SIZE = 100 