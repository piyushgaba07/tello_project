import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
from PIL import Image, ImageTk
import cv2
import threading
import queue
import time
import os
import sys
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import platform  # Import platform to check operating system
from functools import partial

# Import the existing Tello drone functionality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Ensure we can import from the same directory
from drone_control import tello, gesture_model, gesture_labels, process_vlm_input, color_correct_drone_frame
from drone_control import WHISPER_MODEL, AUDIO_DEVICE_INDEX, PHRASE_TIME_LIMIT, COMMANDS

# Constants
LOG_LEVELS = {
    "INFO": "#0066cc",     # Blue
    "ERROR": "#cc0000",    # Red
    "COMMAND": "#009933",  # Green
    "GESTURE": "#9933cc",  # Purple
    "SPEECH": "#cc6600",   # Orange
    "VLM": "#663300",      # Brown
    "DEBUG": "#666666"     # Gray
}

# Style constants
BUTTON_BG = "#f0f0f0"
HEADER_BG = "#e1e1e1"
FRAME_BG = "#f5f5f5"
ACCENT_COLOR = "#3498db"

# Define a custom print function to redirect to GUI
_original_print = print

def custom_print(*args, **kwargs):
    """Replacement for the built-in print function that logs to GUI."""
    # Call the original print for console output
    _original_print(*args, **kwargs)
    
    # If we have a GUI instance, log the message there too
    if hasattr(custom_print, 'gui_instance') and custom_print.gui_instance:
        message = " ".join(map(str, args))
        # Determine level based on content
        level = "INFO"
        if "error" in message.lower() or "exception" in message.lower():
            level = "ERROR"
        elif "command" in message.lower() or "executing" in message.lower():
            level = "COMMAND"
        elif "gesture" in message.lower():
            level = "GESTURE"
        elif "whisper" in message.lower() or "audio" in message.lower() or "speech" in message.lower():
            level = "SPEECH"
        elif "vlm" in message.lower() or "llava" in message.lower():
            level = "VLM"
        
        custom_print.gui_instance.log(message, level)

# Replace the built-in print
print = custom_print

class DroneApp:
    def __init__(self, root):
        """Initialize the Drone Control GUI application."""
        self.root = root
        self.root.title("Tello Drone Control")
        self.root.geometry("1280x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.configure(background=FRAME_BG)
        
        # Set up custom print to log to GUI
        custom_print.gui_instance = self
        
        # App state
        self.drone_in_air = False
        self.active_modality = "idle"  # Set IDLE as default mode
        self.active_camera = "pc"
        self.command_queue = queue.Queue(maxsize=5)
        self.frame_queue = queue.Queue(maxsize=1)
        self.log_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.running = True
        self.last_vlm_command_time = 0
        self.vlm_cooldown = 3
        self.active_buttons = {}  # Track active buttons for highlighting
        
        # For gesture detection
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.pred_gesture = ""
        self.temp_gesture = ""
        self.gesture_count = 0
        
        # For speech recognition
        self.recognizer = sr.Recognizer()
        self.speech_thread = None
        
        # Initialize PC camera
        self.pc_cam = cv2.VideoCapture(0)
        if not self.pc_cam.isOpened():
            messagebox.showwarning("Camera Warning", "Could not open PC camera. Defaulting to drone camera.")
            self.active_camera = "drone"
        
        # Custom styling
        self.custom_font = font.nametofont("TkDefaultFont").copy()
        self.custom_font.configure(size=10)
        self.header_font = font.Font(family="Helvetica", size=12, weight="bold")
        
        # Setup UI components
        self.setup_ui()
        
        # Bind keyboard controls
        self.setup_keyboard_controls()
        
        # Start threads
        self.start_threads()
        
        # Update UI periodically
        self.update_ui()
        self.log("Tello Drone Control GUI started", "INFO")
        self.log(f"Battery level: {tello.get_battery()}%", "INFO")
        
    def setup_ui(self):
        """Create the user interface with improved styling."""
        # Apply a theme
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
            
        # Make all button clicks more responsive
        self.root.option_add('*Button.TakeFocus', 1)
        
        # Platform-specific UI optimizations
        if platform.system() == 'Darwin':  # Only for macOS
            try:
                # This is a macOS-specific command
                self.root.tk.eval('::tk::mac::CGAntialiasLimit 1')
            except tk.TclError:
                # Ignore if command is not available
                pass
                
        # Set consistent scaling for all platforms
        self.root.tk.call('tk', 'scaling', 1.0)
        
        # Configure styles
        style.configure("TFrame", background=FRAME_BG)
        style.configure("TLabel", background=FRAME_BG, font=self.custom_font)
        style.configure("TButton", font=self.custom_font)
        style.configure("Header.TLabel", font=self.header_font, background=HEADER_BG)
        style.configure("TLabelframe", background=FRAME_BG)
        style.configure("TLabelframe.Label", font=self.header_font, background=HEADER_BG)
        style.configure("Control.TButton", background=ACCENT_COLOR)
        style.configure("Active.TButton", background="#ff9900", foreground="white")  # Orange highlight for active buttons
        style.configure("VLM.TButton", background="#4CAF50", foreground="white", font=self.custom_font)
        style.configure("VLM.TFrame", background="#f0f7ff", relief="ridge")
        style.configure("VLMHeader.TLabel", font=self.header_font, background="#0066cc", foreground="white")
        
        # Create a map to make buttons visually respond to clicks
        style.map("TButton",
                  background=[("active", "#d0d0d0"), ("pressed", "#b0b0b0")],
                  relief=[("pressed", "sunken")])
        style.map("Control.TButton",
                  background=[("active", "#2980b9"), ("pressed", "#2471a3")],
                  foreground=[("active", "white")])
        style.map("VLM.TButton",
                  background=[("active", "#45a049"), ("pressed", "#388e3c")],
                  foreground=[("active", "white"), ("pressed", "white")])
        
        # Main container with two frames
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL, style="TPanedwindow")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for video feed and controls
        left_frame = ttk.Frame(main_container, style="TFrame")
        main_container.add(left_frame, weight=3)
        
        # Right frame for logs and status
        right_frame = ttk.Frame(main_container, style="TFrame")
        main_container.add(right_frame, weight=2)
        
        # --- Left Frame Components ---
        # App title
        title_label = ttk.Label(left_frame, text="Tello Drone Control", style="Header.TLabel")
        title_label.pack(fill=tk.X, pady=(0, 10))
        
        # Create top control section
        top_control_frame = ttk.Frame(left_frame, style="TFrame")
        top_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Split for video feed and VLM analysis side by side
        video_vlm_frame = ttk.PanedWindow(left_frame, orient=tk.HORIZONTAL)
        video_vlm_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video feed
        video_frame = ttk.LabelFrame(video_vlm_frame, text="Camera Feed", style="TLabelframe")
        video_vlm_frame.add(video_frame, weight=3)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # VLM command input in the same row as video feed
        vlm_frame = ttk.LabelFrame(video_vlm_frame, text="Vision AI Analysis", style="TLabelframe")
        video_vlm_frame.add(vlm_frame, weight=2)
        
        # Make the label more noticeable with better color
        vlm_info = ttk.Label(vlm_frame, text="Ask the AI about what it sees in the camera feed:", 
                           style="TLabel", wraplength=300, font=self.header_font, foreground="#0066cc")
        vlm_info.pack(fill=tk.X, padx=5, pady=(10, 10))
        
        # Input row with larger entry field
        vlm_input_frame = ttk.Frame(vlm_frame, style="TFrame") 
        vlm_input_frame.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # Larger entry field with specified height
        self.vlm_entry = ttk.Entry(vlm_input_frame, font=self.custom_font)
        self.vlm_entry.pack(fill=tk.X, expand=True, padx=5, pady=8, ipady=4)  # Increased height with ipady
        
        # More noticeable button
        self.send_vlm_btn = ttk.Button(vlm_input_frame, text="Ask AI", command=self.send_vlm_command, 
                                      style="VLM.TButton", width=15)  # Fixed width
        self.send_vlm_btn.pack(padx=8, pady=8)
        
        # Add VLM response display
        vlm_response_frame = ttk.Frame(vlm_frame, style="VLM.TFrame")
        vlm_response_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        response_header = ttk.Label(vlm_response_frame, text="AI Response", 
                                  style="VLMHeader.TLabel", anchor="center")
        response_header.pack(fill=tk.X, padx=0, pady=0)
        
        # Text widget for VLM responses with scrollbar
        vlm_response_container = ttk.Frame(vlm_response_frame, style="TFrame")
        vlm_response_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.vlm_response_text = scrolledtext.ScrolledText(vlm_response_container, wrap=tk.WORD, 
                                                       height=6, width=40, font=self.custom_font,
                                                       background="#f9f9ff", foreground="#333333",
                                                       borderwidth=1, relief="sunken")
        self.vlm_response_text.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.vlm_response_text.config(state=tk.DISABLED)  # Read-only initially
        
        # Control buttons section
        control_frame = ttk.LabelFrame(left_frame, text="Controls", style="TLabelframe")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Modality controls
        modality_frame = ttk.Frame(control_frame, style="TFrame")
        modality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(modality_frame, text="Control Mode:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        # Modality radio buttons
        self.modality_var = tk.StringVar(value="idle")
        ttk.Radiobutton(modality_frame, text="Gesture", variable=self.modality_var, 
                       value="gesture", command=self.change_modality).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(modality_frame, text="Voice", variable=self.modality_var,
                       value="audio", command=self.change_modality).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(modality_frame, text="Vision Analysis", variable=self.modality_var,
                       value="vlm", command=self.change_modality).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(modality_frame, text="Idle", variable=self.modality_var,
                       value="idle", command=self.change_modality).pack(side=tk.LEFT, padx=5)
        
        # Camera source
        camera_frame = ttk.Frame(control_frame, style="TFrame")
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(camera_frame, text="Camera Source:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.camera_var = tk.StringVar(value=self.active_camera)
        ttk.Radiobutton(camera_frame, text="Local PC", variable=self.camera_var,
                       value="pc", command=self.change_camera).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(camera_frame, text="Tello Drone", variable=self.camera_var,
                       value="drone", command=self.change_camera).pack(side=tk.LEFT, padx=5)
        
        # Basic drone commands (takeoff/land)
        flight_frame = ttk.Frame(control_frame, style="TFrame")
        flight_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.takeoff_btn = ttk.Button(flight_frame, text="Take Off", command=lambda: self.execute_command("takeoff"), style="Control.TButton")
        self.takeoff_btn.pack(side=tk.LEFT, padx=5)
        
        self.land_btn = ttk.Button(flight_frame, text="Land", command=lambda: self.execute_command("land"), style="Control.TButton")
        self.land_btn.pack(side=tk.LEFT, padx=5)
        
        self.hover_btn = ttk.Button(flight_frame, text="Hover", command=lambda: self.execute_command("hover"), style="Control.TButton")
        self.hover_btn.pack(side=tk.LEFT, padx=5)
        
        # Directional controls
        dir_control_frame = ttk.LabelFrame(left_frame, text="Directional Controls (or use Arrow Keys)", style="TLabelframe")
        dir_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a grid for directional buttons
        dir_grid = ttk.Frame(dir_control_frame, style="TFrame")
        dir_grid.pack(padx=10, pady=10)
        
        # Up button (row 0, column 1)
        up_btn = ttk.Button(dir_grid, text="Up ↑", command=lambda: self.execute_command("move_up"))
        up_btn.grid(row=0, column=1, padx=5, pady=5)
        self.active_buttons["move_up"] = up_btn
        
        # Left button (row 1, column 0)
        left_btn = ttk.Button(dir_grid, text="← Left", command=lambda: self.execute_command("move_left"))
        left_btn.grid(row=1, column=0, padx=5, pady=5)
        self.active_buttons["move_left"] = left_btn
        
        # Down button (row 2, column 1)
        down_btn = ttk.Button(dir_grid, text="Down ↓", command=lambda: self.execute_command("move_down"))
        down_btn.grid(row=2, column=1, padx=5, pady=5)
        self.active_buttons["move_down"] = down_btn
        
        # Right button (row 1, column 2)
        right_btn = ttk.Button(dir_grid, text="Right →", command=lambda: self.execute_command("move_right"))
        right_btn.grid(row=1, column=2, padx=5, pady=5)
        self.active_buttons["move_right"] = right_btn
        
        # Forward button (row 1, column 1)
        forward_btn = ttk.Button(dir_grid, text="Forward", command=lambda: self.execute_command("move_forward"))
        forward_btn.grid(row=1, column=1, padx=5, pady=5)
        self.active_buttons["move_forward"] = forward_btn
        
        # Add rotation buttons in a separate row
        rot_frame = ttk.Frame(dir_control_frame, style="TFrame")
        rot_frame.pack(padx=10, pady=(0, 10))
        
        rot_left_btn = ttk.Button(rot_frame, text="Rotate Left (Q)", command=lambda: self.execute_command("rotate_counter_clockwise"))
        rot_left_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["rotate_counter_clockwise"] = rot_left_btn
        
        backward_btn = ttk.Button(rot_frame, text="Backward", command=lambda: self.execute_command("move_back"))
        backward_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["move_back"] = backward_btn
        
        rot_right_btn = ttk.Button(rot_frame, text="Rotate Right (E)", command=lambda: self.execute_command("rotate_clockwise"))
        rot_right_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["rotate_clockwise"] = rot_right_btn
        
        # Add hover button to the frame
        hover_btn = ttk.Button(rot_frame, text="Hover (Space)", command=lambda: self.execute_command("hover"))
        hover_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["hover"] = hover_btn
        
        # Flip controls with button references
        flip_frame = ttk.Frame(dir_control_frame, style="TFrame")
        flip_frame.pack(padx=10, pady=(0, 10))
        
        ttk.Label(flip_frame, text="Flip:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        flip_f_btn = ttk.Button(flip_frame, text="Forward", command=lambda: self.execute_command("flip_f"))
        flip_f_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["flip_f"] = flip_f_btn
        
        flip_b_btn = ttk.Button(flip_frame, text="Backward", command=lambda: self.execute_command("flip_b"))
        flip_b_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["flip_b"] = flip_b_btn
        
        flip_l_btn = ttk.Button(flip_frame, text="Left", command=lambda: self.execute_command("flip_l"))
        flip_l_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["flip_l"] = flip_l_btn
        
        flip_r_btn = ttk.Button(flip_frame, text="Right", command=lambda: self.execute_command("flip_r"))
        flip_r_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["flip_r"] = flip_r_btn
        
        # Store references to takeoff/land buttons
        self.active_buttons["takeoff"] = self.takeoff_btn
        self.active_buttons["land"] = self.land_btn
        
        # Add a keyboard shortcuts help label
        key_help = ttk.Label(dir_control_frame, text="Keyboard: Arrows for direct movement, Q/E for rotation, Space to hover, T for takeoff, L for landing", 
                           style="TLabel", wraplength=600)
        key_help.pack(fill=tk.X, padx=5, pady=5)
        
        # Add explanation of direct API commands
        api_help = ttk.Label(dir_control_frame, text="Commands use direct API methods: move_forward, move_back, move_left, move_right, move_up, move_down, etc.", 
                           style="TLabel", foreground="#cc0000", wraplength=600)
        api_help.pack(fill=tk.X, padx=5, pady=5)
        
        # --- Right Frame Components ---
        # Status section
        status_frame = ttk.LabelFrame(right_frame, text="Status", style="TLabelframe")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Battery status
        battery_frame = ttk.Frame(status_frame, style="TFrame")
        battery_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(battery_frame, text="Battery:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.battery_var = tk.StringVar(value="-- %")
        ttk.Label(battery_frame, textvariable=self.battery_var, style="TLabel").pack(side=tk.LEFT, padx=5)
        
        # Flight status
        flight_frame = ttk.Frame(status_frame, style="TFrame")
        flight_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(flight_frame, text="Flight Status:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.flight_status_var = tk.StringVar(value="Landed")
        ttk.Label(flight_frame, textvariable=self.flight_status_var, style="TLabel").pack(side=tk.LEFT, padx=5)
        
        # Current gesture/command
        current_frame = ttk.Frame(status_frame, style="TFrame")
        current_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(current_frame, text="Current Command:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.current_command_var = tk.StringVar(value="None")
        ttk.Label(current_frame, textvariable=self.current_command_var, style="TLabel").pack(side=tk.LEFT, padx=5)
        
        # Current modality
        mode_frame = ttk.Frame(status_frame, style="TFrame")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Active Mode:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.active_mode_var = tk.StringVar(value="IDLE")
        ttk.Label(mode_frame, textvariable=self.active_mode_var, style="TLabel").pack(side=tk.LEFT, padx=5)
        
        # Log section
        log_frame = ttk.LabelFrame(right_frame, text="Activity Log", style="TLabelframe")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons for log
        log_control_frame = ttk.Frame(log_frame, style="TFrame")
        log_control_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        clear_log_btn = ttk.Button(log_control_frame, text="Clear Log", command=self.clear_log)
        clear_log_btn.pack(side=tk.RIGHT, padx=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20, font=self.custom_font)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.tag_config("INFO", foreground=LOG_LEVELS["INFO"])
        self.log_text.tag_config("ERROR", foreground=LOG_LEVELS["ERROR"])
        self.log_text.tag_config("COMMAND", foreground=LOG_LEVELS["COMMAND"])
        self.log_text.tag_config("GESTURE", foreground=LOG_LEVELS["GESTURE"])
        self.log_text.tag_config("SPEECH", foreground=LOG_LEVELS["SPEECH"])
        self.log_text.tag_config("VLM", foreground=LOG_LEVELS["VLM"])
        self.log_text.tag_config("DEBUG", foreground=LOG_LEVELS["DEBUG"])
        
    def setup_keyboard_controls(self):
        """Set up keyboard bindings for drone control."""
        # Arrow keys for directional movement
        self.root.bind("<Up>", lambda e: self.handle_key_press("move_forward"))
        self.root.bind("<Down>", lambda e: self.handle_key_press("move_back"))
        self.root.bind("<Left>", lambda e: self.handle_key_press("move_left"))
        self.root.bind("<Right>", lambda e: self.handle_key_press("move_right"))
        
        # WASD for forward/backward/left/right
        self.root.bind("w", lambda e: self.handle_key_press("move_forward"))
        self.root.bind("s", lambda e: self.handle_key_press("move_back"))
        self.root.bind("a", lambda e: self.handle_key_press("move_left"))
        self.root.bind("d", lambda e: self.handle_key_press("move_right"))
        
        # QE for rotation
        self.root.bind("q", lambda e: self.handle_key_press("rotate_counter_clockwise"))
        self.root.bind("e", lambda e: self.handle_key_press("rotate_clockwise"))
        
        # RF for up/down
        self.root.bind("r", lambda e: self.handle_key_press("move_up"))
        self.root.bind("f", lambda e: self.handle_key_press("move_down"))
        
        # Space for hover
        self.root.bind("<space>", lambda e: self.handle_key_press("hover"))
        
        # T for takeoff
        self.root.bind("t", lambda e: self.handle_key_press("takeoff"))
        
        # L for land
        self.root.bind("l", lambda e: self.handle_key_press("land"))
        
    def handle_key_press(self, command):
        """Handle a keyboard command."""
        # Only process if not in VLM or idle mode
        if self.active_modality not in ["vlm", "idle"]:
            self.log(f"Keyboard command: {command}", "COMMAND")
            self.execute_command(command)
            return "break"  # Prevent default behavior
        elif self.active_modality == "idle":
            self.log(f"Command ignored (Idle mode): {command}", "INFO")
            return "break"
            
    def log(self, message, level="INFO"):
        """Add a message to the log queue."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put((timestamp, message, level))
        
    def clear_log(self):
        """Clear the log text widget."""
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared", "INFO")
        
    def update_logs(self):
        """Update the log text widget with new messages from the queue."""
        try:
            while not self.log_queue.empty():
                timestamp, message, level = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, f"[{timestamp}] ", level)
                self.log_text.insert(tk.END, f"{message}\n", level)
                self.log_text.see(tk.END)  # Scroll to the end
                self.log_queue.task_done()
        except queue.Empty:
            pass
        
    def update_ui(self):
        """Periodically update the UI components."""
        if self.running:
            try:
                # Update battery status
                try:
                    battery = tello.get_battery()
                    self.battery_var.set(f"{battery} %")
                except Exception as e:
                    self.log(f"Error getting battery: {e}", "ERROR")
                
                # Update flight status
                self.flight_status_var.set("Flying" if self.drone_in_air else "Landed")
                
                # Update active mode display
                self.active_mode_var.set(self.active_modality.upper())
                
                # Update logs
                self.update_logs()
                
                # Schedule the next update
                self.root.after(1000, self.update_ui)
            except Exception as e:
                self.log(f"Error updating UI: {e}", "ERROR")
                self.root.after(1000, self.update_ui)
        
    def update_video_feed(self):
        """Update the video feed display and process gestures if in gesture mode."""
        if not self.running:
            return
            
        # Get frame based on active camera source
        if self.active_camera == "pc":
            ret, frame = self.pc_cam.read()
            if not ret:
                self.log("Error reading PC camera frame, switching to drone camera", "ERROR")
                self.active_camera = "drone"
                self.camera_var.set("drone")
                self.root.after(100, self.update_video_feed)
                return
            frame = cv2.flip(frame, 1)  # Mirror PC camera for better UX
        else:  # active_camera == "drone"
            try:
                frame = tello.get_frame_read().frame
                if frame is None:
                    self.log("Error getting drone camera frame, trying again", "ERROR")
                    self.root.after(100, self.update_video_feed)
                    return
                # Apply color correction to drone feed
                frame = color_correct_drone_frame(frame)
            except Exception as e:
                self.log(f"Error reading drone camera: {e}", "ERROR")
                self.root.after(100, self.update_video_feed)
                return
        
        # Keep latest frame in queue for VLM
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
        else:
            try: 
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame.copy())
            except queue.Empty: pass
            except queue.Full: pass
            
        # Process gestures if in gesture mode and model is available
        display_frame = frame.copy()
        if self.active_modality == "gesture" and gesture_model:
            self.process_gestures(frame, display_frame)
        
        # Convert frame for display
        try:
            # Draw mode and other info on frame
            cv2.putText(display_frame, f"Mode: {self.active_modality.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Camera: {self.active_camera.upper()}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add battery info
            try:
                battery = tello.get_battery()
                cv2.putText(display_frame, f"Battery: {battery}%", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except:
                pass
                
            # Add gesture info if in gesture mode
            if self.active_modality == "gesture" and gesture_model:
                cv2.putText(display_frame, f"Gesture: {self.pred_gesture}", (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert to PIL format and then to ImageTk
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Resize to fit in the UI
            width, height = 640, 480  # Default size
            pil_img = pil_img.resize((width, height), Image.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            self.log(f"Error updating video feed: {e}", "ERROR")
        
        # Schedule next update
        self.root.after(33, self.update_video_feed)  # ~30 fps
        
    def process_gestures(self, frame, display_frame):
        """Process hand gestures in the frame."""
        try:
            # Convert to RGB for MediaPipe
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb_img)
            
            y, x, _ = frame.shape
            landmarks = []
            
            if res.multi_hand_landmarks:
                for handslms in res.multi_hand_landmarks:
                    # Draw landmarks on the display frame
                    self.mp_draw.draw_landmarks(display_frame, handslms, self.mpHands.HAND_CONNECTIONS)
                    
                    # Extract landmark coordinates
                    for lm in handslms.landmark:
                        landmarks.append(lm.x * x)
                        landmarks.append(lm.y * y)
                        
                # Predict gesture if we have enough landmarks
                if landmarks and gesture_model:
                    try:
                        # Make prediction
                        prob = gesture_model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
                        className = np.argmax(prob)
                        
                        # Check confidence
                        if prob[0][className] > 0.9:  # Confidence threshold
                            gesture = gesture_labels.get(className, "unknown")
                            
                            # Count consecutive detections of same gesture
                            if gesture == self.temp_gesture:
                                self.gesture_count += 1
                            else:
                                self.temp_gesture = gesture
                                self.gesture_count = 0
                                
                            # Execute if same gesture detected for multiple frames
                            if self.gesture_count == 5:  # Require 5 consecutive frames
                                self.pred_gesture = self.temp_gesture
                                self.gesture_count = 0
                                self.log(f"Gesture detected: {self.pred_gesture}", "GESTURE")
                                
                                # Execute gesture command
                                self.execute_gesture_command(self.pred_gesture)
                                
                    except Exception as e:
                        self.log(f"Gesture prediction error: {e}", "ERROR")
            else:
                # No hands detected, stop movement if in gesture mode
                if self.active_modality == "gesture" and self.drone_in_air:
                    tello.send_rc_control(0, 0, 0, 0)
                    
        except Exception as e:
            self.log(f"Error processing gestures: {e}", "ERROR")
            
    def execute_gesture_command(self, gesture):
        """Execute a command based on the detected gesture using direct movement methods."""
        try:
            if not self.drone_in_air and gesture != "land":
                # Only allow takeoff if not flying
                if gesture == "forward":  # Use "forward" gesture to takeoff
                    self.execute_command("takeoff")
                else:
                    self.log("Drone is not flying. Use forward gesture to take off.", "GESTURE")
                return
                
            # Execute command based on gesture using direct movement methods
            command = None
            
            if gesture == "forward":
                command = "move_forward"
                tello.move_forward(30)  # Use direct movement instead of RC control
                self.log("Moving forward", "GESTURE")
            elif gesture == "backward":
                command = "move_back"
                tello.move_back(30)  # Use direct movement instead of RC control
                self.log("Moving backward", "GESTURE")
            elif gesture == "right":
                command = "move_right"
                tello.move_right(30)  # Use direct movement instead of RC control
                self.log("Moving right", "GESTURE")
            elif gesture == "left":
                command = "move_left"
                tello.move_left(30)  # Use direct movement instead of RC control
                self.log("Moving left", "GESTURE")
            elif gesture == "up":
                command = "move_up"
                tello.move_up(30)  # Use direct movement instead of RC control
                self.log("Moving up", "GESTURE")
            elif gesture == "down":
                command = "move_down"
                tello.move_down(30)  # Use direct movement instead of RC control
                self.log("Moving down", "GESTURE")
            elif gesture == "flip":
                command = "flip_f"
                tello.flip("f")
                self.log("Performing flip", "GESTURE")
            elif gesture == "land":
                command = "land"
                self.execute_command("land")
                self.log("Landing drone", "GESTURE")
                
            # Update UI to show active command
            if command:
                self.current_command_var.set(command)
                self.highlight_active_button(command)
                
            # Don't need reset_gesture_rc anymore since we're not using continuous RC control
            # But we'll still run it for safety to ensure any pending RC commands are stopped
            self.root.after(500, self.reset_gesture_rc)
            
        except Exception as e:
            self.log(f"Error executing gesture command: {e}", "ERROR")
            
    def reset_gesture_rc(self):
        """Reset RC control after gesture action to prevent continuous movement."""
        if self.active_modality == "gesture" and self.drone_in_air:
            try:
                # Just send a hover command to ensure the drone is stable
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as e:
                self.log(f"Error resetting RC: {e}", "ERROR")
        
    def change_modality(self):
        """Handle modality change with immediate feedback."""
        new_modality = self.modality_var.get()
        if new_modality != self.active_modality:
            # Immediately update UI to show the change is registered
            self.log(f"Switching to {new_modality.upper()} mode", "INFO")
            
            # Special case for gesture mode
            if new_modality == "gesture" and not gesture_model:
                messagebox.showwarning("Modality Warning", "Gesture model not loaded. Cannot switch to gesture mode.")
                self.modality_var.set(self.active_modality)  # Revert selection
                return
                
            # Force immediate UI update for feedback
            self.root.update_idletasks()
            
            # Set status display
            self.current_command_var.set(f"Switched to {new_modality.upper()} mode")
                
            # Update the actual mode
            self.active_modality = new_modality
            self.active_mode_var.set(new_modality.upper())
            
            # Clear VLM response when switching modes
            if new_modality != "vlm":
                self.vlm_response_text.config(state=tk.NORMAL)
                self.vlm_response_text.delete(1.0, tk.END)
                self.vlm_response_text.config(state=tk.DISABLED)
            
            # Stop any ongoing movement when changing modes
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as e:
                self.log(f"Error stopping movement: {e}", "ERROR")
                
    def change_camera(self):
        """Handle camera source change with immediate feedback."""
        new_camera = self.camera_var.get()
        if new_camera != self.active_camera:
            # Immediately update UI to show the change is registered
            self.log(f"Switching to {new_camera.upper()} camera", "INFO")
            self.active_camera = new_camera
            
            # Force an immediate update of the display
            self.root.update_idletasks()
            
            # Visual feedback on status
            if new_camera == "pc":
                self.current_command_var.set("Switched to PC camera")
                # Verify PC camera is working and open it if not
                if not self.pc_cam.isOpened():
                    self.log("Reopening PC camera...", "INFO")
                    try:
                        # Release any existing camera first
                        if hasattr(self, 'pc_cam') and self.pc_cam is not None:
                            self.pc_cam.release()
                            
                        # Open camera with a short timeout
                        self.pc_cam = cv2.VideoCapture(0)
                        
                        # Check if opened successfully
                        if not self.pc_cam.isOpened():
                            self.log("PC camera not available, reverting to DRONE camera.", "ERROR")
                            self.active_camera = "drone"
                            self.camera_var.set("drone")  # Update radio button
                            return
                            
                        self.log("PC camera opened successfully", "INFO")
                    except Exception as e:
                        self.log(f"Error opening PC camera: {e}", "ERROR")
                        self.active_camera = "drone"
                        self.camera_var.set("drone")  # Update radio button
                        return
            else:  # drone camera
                self.current_command_var.set("Switched to Drone camera")
                
            # Ensure we get a fresh frame immediately
            self.root.after(10, self.update_video_feed)
            
    def execute_command(self, command):
        """Execute a drone command with immediate visual feedback."""
        # Don't execute any commands in idle mode
        if self.active_modality == "idle":
            self.log(f"Command ignored (Idle mode): {command}", "INFO")
            return
        
        try:
            # Immediate visual feedback - update UI first before any processing
            self.current_command_var.set(command)
            self.log(f"Executing command: {command}", "COMMAND")
            
            # Force immediate UI update
            self.root.update_idletasks()
            
            # Highlight the active button
            self.highlight_active_button(command)
            
            # Process the actual command
            if command == "takeoff":
                if not self.drone_in_air:
                    self.log("🚀 Taking off...", "COMMAND")
                    tello.takeoff()
                    self.drone_in_air = True
                else:
                    self.log("Drone already in air.", "INFO")
            elif command == "land":
                if self.drone_in_air:
                    self.log("🛬 Landing...", "COMMAND")
                    tello.land()
                    self.drone_in_air = False
                else:
                    self.log("Drone already on ground.", "INFO")
            elif self.drone_in_air:  # Most commands only work if flying
                if command == "move_forward":
                    tello.move_forward(30)
                elif command == "move_back":
                    tello.move_back(30)
                elif command == "rotate_counter_clockwise":
                    tello.rotate_counter_clockwise(45)
                elif command == "rotate_clockwise":
                    tello.rotate_clockwise(45)
                elif command == "move_left":
                    tello.move_left(30)
                elif command == "move_right":
                    tello.move_right(30)
                elif command == "move_up":
                    tello.move_up(30)
                elif command == "move_down":
                    tello.move_down(30)
                elif command == "hover":
                    self.log("⏸️ Hovering...", "COMMAND")
                    tello.send_rc_control(0, 0, 0, 0)  # Explicit hover
                elif command == "flip_f":
                    tello.flip("f")
                elif command == "flip_b":
                    tello.flip("b")
                elif command == "flip_l":
                    tello.flip("l")
                elif command == "flip_r":
                    tello.flip("r")
                else:
                    self.log(f"❌ Command '{command}' known but not executable in current state.", "ERROR")
            elif command not in ["takeoff", "land"]:
                self.log(f"Cannot execute '{command}' while landed. Try 'takeoff'.", "ERROR")

            # Schedule button unhighlighting after a delay
            self.root.after(500, lambda: self.reset_button_highlight(command))

        except Exception as e:
            self.log(f"Error executing command '{command}': {e}", "ERROR")
            # Attempt to stabilize after error if in air
            if self.drone_in_air:
                try:
                    self.log("Attempting to hover after error...", "COMMAND")
                    tello.send_rc_control(0, 0, 0, 0)
                except Exception as hover_e:
                    self.log(f"Could not stabilize after error: {hover_e}", "ERROR")
            
            # Reset button highlight after error
            self.reset_button_highlight(command)
            
    def highlight_active_button(self, command):
        """Highlight the button corresponding to the active command with immediate feedback."""
        if command in self.active_buttons:
            button = self.active_buttons[command]
            button.configure(style="Active.TButton")
            # Force immediate visual update
            self.root.update_idletasks()
            
    def reset_button_highlight(self, command):
        """Reset the button highlight after a command completes."""
        if command in self.active_buttons:
            button = self.active_buttons[command]
            if "flip" in command:
                button.configure(style="TButton")
            elif command in ["takeoff", "land", "hover"]:
                button.configure(style="Control.TButton")
            else:
                button.configure(style="TButton")
        
    def start_threads(self):
        """Start the required background threads."""
        self.threads = []
        
        # Command processor thread
        cmd_thread = threading.Thread(
            target=self.command_processor,
            name="CommandProcessor",
            daemon=True
        )
        self.threads.append(cmd_thread)
        cmd_thread.start()
        
        # Voice listener thread 
        speech_thread = threading.Thread(
            target=self.listen_for_voice_commands,
            name="VoiceListener",
            daemon=True
        )
        self.threads.append(speech_thread)
        speech_thread.start()
        
        # Start video feed update in the main thread using after()
        self.update_video_feed()
        
    def on_close(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.log("Shutting down application...", "INFO")
            self.running = False
            self.stop_event.set()
            
            # Land drone if in air
            if self.drone_in_air:
                try:
                    self.log("Landing drone before exit...", "COMMAND")
                    tello.land()
                except Exception as e:
                    self.log(f"Error landing drone: {e}", "ERROR")
            
            # Release camera
            if self.pc_cam.isOpened():
                self.pc_cam.release()
            
            # Stop drone stream
            try:
                tello.streamoff()
            except Exception as e:
                self.log(f"Error stopping stream: {e}", "ERROR")
                
            # Wait for threads to finish (briefly)
            for thread in self.threads:
                if thread.is_alive():
                    thread.join(1.0)  # Wait with timeout
                    
            # Restore original print function
            global print
            print = _original_print
                
            self.root.destroy()
            
    def process_audio_command(self, command_text):
        """Process a command from speech recognition."""
        self.log(f"Processing voice command: '{command_text}'", "SPEECH")
        
        # Recognize the command
        drone_command = self.recognize_command(command_text)
        if drone_command not in ["Unknown Command", "Already Landed", "Already Flying"]:
            self.execute_command(drone_command)
        else:
            self.log(f"Voice command ignored: {drone_command}", "SPEECH")
            
    def process_vlm_command(self, command_text):
        """Process a command from VLM without controlling the drone."""
        current_time = time.time()
        if current_time - self.last_vlm_command_time > self.vlm_cooldown:
            self.log(f"Processing VLM query: '{command_text}'", "VLM")
            
            # Get the latest frame for processing
            frame = None
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                self.log("No frame available for VLM processing", "ERROR")
                return
                
            self.last_vlm_command_time = current_time
            
            # Process the VLM command with the current frame
            vlm_response = process_vlm_input(command_text, frame)
            
            # Display the response in the log
            self.log(f"VLM response: {vlm_response}", "VLM")
            self.current_command_var.set("VLM Analysis")
            
            # Update the VLM response text widget with better formatting
            self.vlm_response_text.config(state=tk.NORMAL)
            self.vlm_response_text.delete(1.0, tk.END)
            
            # Format the response with timestamp
            timestamp = time.strftime("%H:%M:%S")
            formatted_response = f"[{timestamp}] Query: {command_text}\n\n{vlm_response}"
            
            self.vlm_response_text.insert(tk.END, formatted_response)
            
            # Add tags for styling different parts
            self.vlm_response_text.tag_configure("timestamp", foreground="#0066cc", font=font.Font(family="Helvetica", size=9, weight="bold"))
            self.vlm_response_text.tag_configure("query", foreground="#009933", font=font.Font(family="Helvetica", size=9, slant="italic"))
            self.vlm_response_text.tag_configure("response", foreground="#333333", font=self.custom_font)
            
            # Apply tags
            self.vlm_response_text.tag_add("timestamp", "1.0", "1.10")
            self.vlm_response_text.tag_add("query", "1.11", "2.0")
            self.vlm_response_text.tag_add("response", "3.0", "end")
            
            self.vlm_response_text.config(state=tk.DISABLED)
            
        else:
            self.log(f"VLM query '{command_text}' skipped due to cooldown", "VLM")
            
    def recognize_command(self, text):
        """Check if recognized text matches any drone command and convert to direct API commands."""
        text = text.lower().strip()
        if not text:
            return "Unknown Command"

        # Command mapping from descriptive terms to API method names
        command_mapping = {
            "take off": "takeoff",
            "takeoff": "takeoff",
            "land": "land",
            "move forward": "move_forward",
            "go forward": "move_forward", 
            "fly forward": "move_forward",
            "forward": "move_forward",
            "move backward": "move_back",
            "go back": "move_back",
            "fly backward": "move_back",
            "backward": "move_back",
            "back": "move_back",
            "turn left": "rotate_counter_clockwise",
            "rotate left": "rotate_counter_clockwise", 
            "turn right": "rotate_clockwise",
            "rotate right": "rotate_clockwise",
            "move left": "move_left",
            "go left": "move_left",
            "fly left": "move_left",
            "left": "move_left", 
            "strafe left": "move_left",
            "move right": "move_right",
            "go right": "move_right", 
            "fly right": "move_right",
            "right": "move_right",
            "strafe right": "move_right",
            "move up": "move_up",
            "ascend": "move_up", 
            "fly up": "move_up",
            "up": "move_up",
            "move down": "move_down",
            "descend": "move_down", 
            "fly down": "move_down",
            "down": "move_down",
            "hover": "hover",
            "stay": "hover", 
            "hold position": "hover",
            "stop moving": "hover",
            "flip forward": "flip_f",
            "flip front": "flip_f",
            "flip backward": "flip_b",
            "flip back": "flip_b",
            "flip left": "flip_l",
            "flip right": "flip_r"
        }

        # Check exact matches first
        for phrase, api_command in command_mapping.items():
            if text == phrase:
                # Handle takeoff/land special cases
                if api_command == "land" and not self.drone_in_air:
                    return "Already Landed"
                if api_command == "takeoff" and self.drone_in_air:
                    return "Already Flying"
                return api_command

        # Check partial matches
        for phrase, api_command in command_mapping.items():
            if phrase in text:
                # Handle takeoff/land special cases
                if api_command == "land" and not self.drone_in_air:
                    return "Already Landed"
                if api_command == "takeoff" and self.drone_in_air:
                    return "Already Flying"
                return api_command

        self.log(f"No matching command found for: '{text}'", "DEBUG")
        return "Unknown Command"
        
    def command_processor(self):
        """Process commands from the queue based on active modality."""
        self.log("Command processor started", "INFO")
        
        while not self.stop_event.is_set():
            try:
                # Get a command from the queue with a timeout
                command_input = self.command_queue.get(timeout=1)
                
                # Process based on active modality
                if self.active_modality == "audio":
                    self.process_audio_command(command_input)
                elif self.active_modality == "vlm":
                    self.process_vlm_command(command_input)
                # Note: gesture commands are handled directly in the video processing
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue  # No command in queue
            except Exception as e:
                self.log(f"Command processing error: {e}", "ERROR")
                # Ensure task_done is called even on error if item was retrieved
                if 'command_input' in locals():
                    try: 
                        self.command_queue.task_done()
                    except ValueError: 
                        pass  # Already marked done
        
        self.log("Command processor stopped", "INFO")
        
    def listen_for_voice_commands(self):
        """Listen for voice commands using Whisper and put them in the queue."""
        self.log(f"Voice listener started (Whisper model: {WHISPER_MODEL})", "SPEECH")
        
        # Initialize recognizer
        r = self.recognizer
        
        # Try initial ambient noise adjustment
        try:
            with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                self.log("Adjusting for ambient noise...", "SPEECH")
                r.adjust_for_ambient_noise(source, duration=2)
                self.log(f"Adjusted energy threshold: {r.energy_threshold}", "SPEECH")
        except Exception as e:
            self.log(f"Error during ambient noise adjustment: {e}", "ERROR")
            self.log("Voice recognition might be unreliable", "ERROR")
            r.energy_threshold = 300  # Set a default
            
        # Main listening loop
        while not self.stop_event.is_set():
            # Only listen if in audio mode
            if self.active_modality != "audio":
                time.sleep(0.5)
                continue
                
            try:
                # Create a new microphone instance each time
                self.log("Listening for command...", "SPEECH")
                with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                    audio = r.listen(source, phrase_time_limit=PHRASE_TIME_LIMIT)
                
                self.log("Processing audio...", "SPEECH")
                command_text = r.recognize_whisper(audio, language="english", model=WHISPER_MODEL)
                command_text = command_text.strip()
                
                if command_text:
                    self.log(f"Recognized: '{command_text}'", "SPEECH")
                    
                    # Add to command queue
                    if not self.command_queue.full():
                        self.command_queue.put(command_text)
                    else:
                        self.log("Command queue full, discarding older command", "SPEECH")
                        try: 
                            self.command_queue.get_nowait()
                        except queue.Empty: 
                            pass
                        try: 
                            self.command_queue.put_nowait(command_text)
                        except queue.Full: 
                            pass
                
            except sr.WaitTimeoutError:
                self.log("No command heard", "SPEECH")
            except sr.UnknownValueError:
                self.log("Could not understand audio", "SPEECH")
            except sr.RequestError as e:
                self.log(f"Error with Whisper service: {e}", "ERROR")
            except Exception as e:
                self.log(f"Voice listening error: {e}", "ERROR")
                time.sleep(1)
                
        self.log("Voice listener stopped", "SPEECH")

    def send_vlm_command(self):
        """Send a query to the Vision AI for analysis with immediate feedback."""
        query = self.vlm_entry.get().strip()
        if query:
            # Provide immediate visual feedback
            if self.active_modality != "vlm":
                self.log("Switching to Vision Analysis mode.", "INFO")
                self.active_modality = "vlm"
                self.modality_var.set("vlm")
                self.active_mode_var.set("VLM")
                
            # Update UI immediately
            self.current_command_var.set(f"Processing: {query}")
            self.root.update_idletasks()
                
            # Process the query
            self.log(f"Sending vision query: {query}", "VLM")
            self.command_queue.put(query)
            self.vlm_entry.delete(0, tk.END)  # Clear the entry

# Main entry point
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = DroneApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Critical error: {e}")
        try:
            # Emergency landing if app crashes
            if hasattr(app, 'drone_in_air') and app.drone_in_air:
                tello.land()
            tello.streamoff()
        except:
            pass  # Best effort only 