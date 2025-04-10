"""
Main application module for the Tello Drone Control GUI.
This integrates all components into a complete application.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font, PhotoImage
import cv2
import threading
import queue
import time
import os
import sys
import platform
from functools import partial
import builtins

# Import our modules
from utils.config import BUTTON_BG, HEADER_BG, FRAME_BG, ACCENT_COLOR, LOG_LEVELS, DEFAULT_MODE
from utils.logger import logger, custom_print

# Import UI components
from gui.ui_components import (
    HeaderLabel, LogDisplay, StatusBar, 
    ControlPanel, VlmPanel
)

# Import handlers
from gui.camera_handler import CameraHandler
from gui.command_handler import CommandHandler
from gui.gesture_processor import GestureProcessor
from gui.speech_handler import SpeechHandler

# Import drone control (existing module)
# Need to import directly from drone_control since it's in the same directory as gui
import sys
import os
# Add parent directory to path so we can import drone_control
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drone_control import (
    tello, gesture_model, gesture_labels, process_vlm_input,
    WHISPER_MODEL, AUDIO_DEVICE_INDEX, PHRASE_TIME_LIMIT
)

class DroneApp(tk.Tk):
    """The main application window for the Tello Drone Control GUI."""
    def __init__(self):
        super().__init__()
        # Window setup
        self.title("Tello Drone Control")
        self.geometry("1280x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize queues before any handlers
        self.command_queue = queue.Queue(maxsize=10)
        self.log_queue = queue.Queue(maxsize=100)
        
        # Thread control flags
        self.keep_alive_active = True
        
        # Initialize UI containers first
        self.setup_ui_containers()
        
        # Create log display early - needed by handlers
        self.log_display = LogDisplay(self.right_frame, height=10)
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Set up the custom print function to use our logger
        # But keep a reference to the original print
        self._original_print = builtins.print
        builtins.print = self.custom_print
        
        # Complete the UI setup
        self.setup_ui_components()
        
        # Initialize handlers AFTER log_display is created
        self.setup_handlers()
        
        self.setup_styles()
        self.setup_keyboard_controls()
        
        # Start the UI updater
        self.after(500, self.update_ui)
        
        # Log initial message to confirm GUI is operational
        print("Tello Drone Control GUI Initialized")

    def setup_ui_containers(self):
        """Set up the main UI containers."""
        # Main container frames
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left and right columns
        self.left_frame = tk.Frame(self.main_frame, width=640)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = tk.Frame(self.main_frame, width=640)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Status bar at the bottom
        self.status_bar = tk.Frame(self, height=20, bg="#f0f0f0")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Status: Initializing...", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.battery_label = tk.Label(self.status_bar, text="Battery: --", anchor=tk.E)
        self.battery_label.pack(side=tk.RIGHT, padx=10)

    def setup_handlers(self):
        """Initialize the various handlers for drone interaction."""
        # Initialize with empty attributes first to avoid "attribute not found" errors
        self.camera_handler = None
        self.command_handler = None
        self.gesture_handler = None
        self.speech_handler = None
        
        try:
            # Initialize camera handler first - needed by gesture handler
            self.camera_handler = CameraHandler(
                drone_video_address="udp://@0.0.0.0:11111",
                camera_idx=0,
                record_video=False,
                overlay_canvas=None
            )
            
            # Initialize command handler - controls actual drone
            self.command_handler = CommandHandler(command_queue=self.command_queue)
            
            # Initialize gesture handler
            self.gesture_handler = GestureProcessor(
                gesture_model=gesture_model,
                gesture_labels=gesture_labels
            )
            
            # Initialize speech handler - at this point log_display is available
            self.speech_handler = SpeechHandler(command_queue=self.command_queue)
            
            print("All handlers initialized successfully")
        except Exception as e:
            self._original_print(f"Error initializing handlers: {e}")
            # Continue with UI setup to avoid crashing completely

    def setup_ui_components(self):
        """Set up the UI components after handlers are initialized."""
        # Initialize attribute storage
        self.active_buttons = {}
        
        # Video feed in left frame
        self.video_label = tk.Label(self.left_frame)
        self.video_label.pack(pady=10)
        
        # Control buttons in right frame
        self.control_frame = tk.Frame(self.right_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Flight controls
        self.flight_frame = tk.LabelFrame(self.control_frame, text="Flight Controls")
        self.flight_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.takeoff_button = tk.Button(
            self.flight_frame, 
            text="Takeoff", 
            command=lambda: self.execute_command("takeoff")
        )
        self.takeoff_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.active_buttons["takeoff"] = self.takeoff_button
        
        self.land_button = tk.Button(
            self.flight_frame, 
            text="Land", 
            command=lambda: self.execute_command("land")
        )
        self.land_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.active_buttons["land"] = self.land_button
        
        self.hover_button = tk.Button(
            self.flight_frame, 
            text="Hover", 
            command=lambda: self.execute_command("hover")
        )
        self.hover_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.active_buttons["hover"] = self.hover_button
        
        # Settings
        self.settings_frame = tk.LabelFrame(self.control_frame, text="Settings")
        self.settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Initialize control variables
        self.voice_var = tk.BooleanVar(value=False)
        self.gesture_var = tk.BooleanVar(value=False)
        self.camera_var = tk.BooleanVar(value=True)
        self.drone_camera_var = tk.BooleanVar(value=False)
        self.modality_var = tk.StringVar(value="idle")
        
        # Toggle voice recognition
        self.voice_checkbox = tk.Checkbutton(
            self.settings_frame,
            text="Voice Control",
            variable=self.voice_var,
            command=self.toggle_voice_recognition
        )
        self.voice_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        
        # Toggle gesture recognition
        self.gesture_checkbox = tk.Checkbutton(
            self.settings_frame,
            text="Gesture Control",
            variable=self.gesture_var,
            command=self.toggle_gesture_recognition
        )
        self.gesture_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        
        # Toggle camera feed
        self.camera_checkbox = tk.Checkbutton(
            self.settings_frame,
            text="Camera Feed",
            variable=self.camera_var,
            command=self.toggle_camera_feed
        )
        self.camera_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        
        # Toggle drone video
        self.drone_camera_checkbox = tk.Checkbutton(
            self.settings_frame,
            text="Drone Camera",
            variable=self.drone_camera_var,
            command=self.toggle_drone_camera
        )
        self.drone_camera_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        
        # Help button
        self.help_button = tk.Button(
            self.settings_frame,
            text="Help",
            command=self.show_help
        )
        self.help_button.pack(anchor=tk.W, padx=5, pady=5)
        
        # Create control panel
        try:
            self.control_panel = ControlPanel(self.right_frame, command_callback=self.execute_command)
            self.control_panel.pack(fill=tk.X, padx=10, pady=10)
        except Exception as e:
            self.safe_print(f"Error creating control panel: {e}")
            # Create an empty frame as placeholder
            self.control_panel = tk.Frame(self.right_frame)
            self.control_panel.pack(fill=tk.X, padx=10, pady=10)
        
        # Create VLM panel
        try:
            self.vlm_panel = VlmPanel(self.right_frame, command_callback=self.send_vlm_command)
            self.vlm_panel.pack(fill=tk.X, padx=10, pady=10)
        except Exception as e:
            self.safe_print(f"Error creating VLM panel: {e}")
            # Create an empty frame as placeholder
            self.vlm_panel = tk.Frame(self.right_frame)
            self.vlm_panel.pack(fill=tk.X, padx=10, pady=10)
        
        # Create status bar
        try:
            self.status_bar = StatusBar(self)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        except Exception as e:
            self.safe_print(f"Error creating status bar: {e}")
            # Use basic status bar as fallback
            self.status_bar = tk.Frame(self, height=20, bg="#f0f0f0")
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Log initial status
        self.log("UI components initialized", "INFO")
    
    def custom_print(self, *args, **kwargs):
        """Custom print function that logs to both console and GUI."""
        # Print to the original console
        self._original_print(*args, **kwargs)
        
        # Format the message
        message = " ".join(str(arg) for arg in args)
        
        # Add to GUI log if log_display exists and is ready
        if hasattr(self, 'log_display') and self.log_display:
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            
            # Add to log queue (thread-safe) if available
            if hasattr(self, 'log_queue') and not self.log_queue.full():
                try:
                    self.log_queue.put(log_message)
                    # Schedule the log to be displayed (from main thread)
                    self.after_idle(self.process_log_queue)
                except Exception:
                    pass  # Silently handle queue errors
    
    def safe_print(self, message):
        """Safe print function that avoids recursion."""
        if hasattr(self, '_original_print'):
            self._original_print(message)
        else:
            print(message)  # Fallback to standard print
    
    def setup_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        
        # Custom fonts
        self.custom_font = font.nametofont("TkDefaultFont").copy()
        self.custom_font.configure(size=10)
        self.header_font = font.Font(family="Helvetica", size=12, weight="bold")
        
        # Configure styles
        self.configure_styles(style)
    
    def configure_styles(self, style):
        """Configure ttk styles for the application."""
        style.configure("TFrame", background=FRAME_BG)
        style.configure("TLabel", background=FRAME_BG, font=self.custom_font)
        style.configure("TButton", font=self.custom_font)
        style.configure("Header.TLabel", font=self.header_font, background=HEADER_BG)
        style.configure("TLabelframe", background=FRAME_BG)
        style.configure("TLabelframe.Label", font=self.header_font, background=HEADER_BG)
        style.configure("Control.TButton", background=ACCENT_COLOR)
        style.configure("Active.TButton", background="#ff9900", foreground="white")
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
    
    def setup_keyboard_controls(self):
        """Set up keyboard shortcuts."""
        # Basic flight controls
        self.bind("t", lambda e: self.handle_key_press("takeoff"))
        self.bind("l", lambda e: self.handle_key_press("land"))
        self.bind("<space>", lambda e: self.handle_key_press("hover"))
        
        # Directional controls
        self.bind("<Up>", lambda e: self.handle_key_press("move_forward"))
        self.bind("<Down>", lambda e: self.handle_key_press("move_back"))
        self.bind("<Left>", lambda e: self.handle_key_press("move_left"))
        self.bind("<Right>", lambda e: self.handle_key_press("move_right"))
        self.bind("w", lambda e: self.handle_key_press("move_up"))
        self.bind("s", lambda e: self.handle_key_press("move_down"))
        self.bind("a", lambda e: self.handle_key_press("rotate_counter_clockwise"))
        self.bind("d", lambda e: self.handle_key_press("rotate_clockwise"))
        
        # Mode keys
        self.bind("1", lambda e: self.set_modality("gesture"))
        self.bind("2", lambda e: self.set_modality("audio"))
        self.bind("3", lambda e: self.set_modality("vlm"))
        self.bind("0", lambda e: self.set_modality("idle"))
        
        # Camera toggle
        self.bind("c", lambda e: self.toggle_camera())
    
    def handle_key_press(self, command):
        """Handle keyboard command."""
        # Only process if not in VLM or idle mode
        if self.command_handler.active_modality not in ["vlm", "idle"]:
            self.log(f"Keyboard command: {command}", "COMMAND")
            self.execute_command(command)
            return "break"  # Prevent default behavior
        elif self.command_handler.active_modality == "idle":
            self.log(f"Command ignored (Idle mode): {command}", "INFO")
            return "break"
    
    def log(self, message, level="INFO"):
        """Add a log message to the log display."""
        self.log_display.add_log(message, level)
    
    def clear_log(self):
        """Clear the log display."""
        self.log_display.clear()
        logger.clear()
    
    def update_ui(self):
        """Update the UI components periodically."""
        # Update battery level
        try:
            if hasattr(self, 'status_bar') and self.status_bar:
                battery = tello.get_battery()
                self.status_bar.update_battery(battery)
        except Exception as e:
            if hasattr(self, 'log_display') and self.log_display:
                self.log(f"Error getting battery level: {e}", "ERROR")
        
        # Update flight status if command_handler exists
        if hasattr(self, 'command_handler') and self.command_handler and hasattr(self, 'status_bar') and self.status_bar:
            try:
                self.status_bar.update_flight_status(
                    "In Flight" if self.command_handler.is_drone_flying() else "Landed"
                )
            except Exception:
                pass
        
        # Update video feed if camera_handler exists
        if hasattr(self, 'camera_handler') and self.camera_handler:
            try:
                self.update_video_feed()
            except Exception as e:
                if hasattr(self, 'log_display') and self.log_display:
                    self.log(f"Error updating video feed: {e}", "ERROR")
        
        # Update logs from queue
        if hasattr(self, 'log_queue') and self.log_queue:
            try:
                self.update_logs()
            except Exception:
                pass
        
        # Schedule next update (every 100ms)
        self.after(100, self.update_ui)
    
    def update_logs(self):
        """Update log display from the log queue."""
        while not self.log_queue.empty():
            try:
                log_entry = self.log_queue.get_nowait()
                self.log(log_entry["message"], log_entry["level"])
            except queue.Empty:
                break
    
    def update_video_feed(self):
        """Update the video feed with current frame and process gestures if active."""
        frame = self.camera_handler.get_frame()
        if frame is None:
            return
        
        # Process the frame for gestures if in gesture mode
        if self.command_handler.active_modality == "gesture":
            gesture, processed_frame = self.gesture_handler.process_frame(frame)
            
            # Execute gesture command if detected
            if gesture:
                self.command_handler.process_gesture_command(gesture)
                
                # Update status with detected gesture
                self.status_bar.update_command(f"Gesture: {gesture}")
                
                # Highlight corresponding control button
                self.highlight_active_button(gesture)
        else:
            # Just use the frame without gesture processing if not in gesture mode
            processed_frame = frame
        
        # Convert to Tkinter-compatible image
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))  # Resize for display
        img = PhotoImage(data=cv2.imencode('.ppm', img)[1].tobytes())
        
        # Update video label
        self.video_label.configure(image=img)
        self.video_label.image = img  # Keep a reference to prevent garbage collection
    
    def highlight_active_button(self, command):
        """Highlight the active button for visual feedback."""
        # Check control panel buttons
        self.control_panel.highlight_button(command)
        
        # Check other buttons
        if command in self.active_buttons:
            self.active_buttons[command].state(['pressed'])
            
        # Schedule button reset
        self.after(500, lambda: self.reset_button_highlight(command))
    
    def reset_button_highlight(self, command):
        """Reset button highlight after a delay."""
        # Reset control panel buttons
        self.control_panel.reset_button(command)
        
        # Reset other buttons
        if command in self.active_buttons:
            self.active_buttons[command].state(['!pressed'])
    
    def start_threads(self):
        """Start background threads."""
        # Start command processor
        self.command_handler.start_processor()
        
        # Start speech recognition if in audio mode
        if self.command_handler.active_modality == "audio":
            self.speech_handler.start_listener()
    
    def change_modality(self):
        """Handle modality change."""
        new_modality = self.modality_var.get()
        current_modality = self.command_handler.active_modality
        
        if new_modality != current_modality:
            # Check if gesture model is available for gesture mode
            if new_modality == "gesture" and not self.gesture_handler.is_available():
                messagebox.showwarning("Modality Warning", "Gesture model not loaded. Cannot switch to gesture mode.")
                self.modality_var.set(current_modality)  # Revert selection
                return
                
            # Log the change
            self.log(f"Switching to {new_modality.upper()} mode", "INFO")
            
            # Update command handler
            self.command_handler.set_modality(new_modality)
            
            # Update status display
            self.status_bar.update_mode(new_modality)
            self.status_bar.update_command(f"Switched to {new_modality.upper()} mode")
            
            # Start/stop speech recognition based on mode
            if new_modality == "audio":
                self.speech_handler.start_listener()
            elif current_modality == "audio":
                self.speech_handler.stop_listener()
                
            # Clear VLM response when switching modes
            if new_modality != "vlm":
                self.vlm_panel.clear_response()
    
    def change_camera(self):
        """Switch between PC and drone camera."""
        new_camera = self.camera_var.get()
        current_camera = self.camera_handler.get_active_camera()
        
        if new_camera != current_camera:
            # Switch camera
            success = self.camera_handler.switch_camera()
            
            if success:
                self.log(f"Switched to {new_camera.upper()} camera", "INFO")
            else:
                # If switch failed, revert the selection
                self.camera_var.set(self.camera_handler.get_active_camera())
                self.log(f"Failed to switch to {new_camera} camera", "ERROR")
    
    def toggle_camera(self):
        """Toggle between PC and drone camera (keyboard shortcut)."""
        current_camera = self.camera_handler.get_active_camera()
        new_camera = "drone" if current_camera == "pc" else "pc"
        self.camera_var.set(new_camera)
        self.change_camera()
    
    def execute_command(self, command):
        """Execute a drone command."""
        success = self.command_handler.execute_command(command)
        if success:
            self.status_bar.update_command(command)
            self.highlight_active_button(command)
    
    def on_command_executed(self, command):
        """Callback when a command is executed."""
        self.status_bar.update_command(command)
        
        # Update flight status if takeoff/land
        if command == "takeoff":
            self.status_bar.update_flight_status("In Flight")
        elif command == "land":
            self.status_bar.update_flight_status("Landed")
    
    def send_vlm_command(self, query):
        """Send a VLM command for processing."""
        # Switch to VLM mode if not already
        if self.command_handler.active_modality != "vlm":
            self.set_modality("vlm")
        
        # Enqueue the command
        self.command_handler.enqueue_command(query)
        
        # Update status
        self.status_bar.update_command(f"Processing: {query}")
    
    def on_vlm_response(self, query, response):
        """Callback when a VLM response is received."""
        # Update the VLM panel with the response
        self.vlm_panel.update_response(query, response)
    
    def set_modality(self, modality):
        """Set the modality directly."""
        self.modality_var.set(modality)
        self.change_modality()
    
    def on_close(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit? This will land the drone if it's flying."):
            # Set flags to stop all threads
            self.keep_alive_active = False
            
            # Stop all threads
            if hasattr(self, 'speech_handler') and self.speech_handler:
                self.speech_handler.stop_listener()
            
            if hasattr(self, 'command_handler') and self.command_handler:
                self.command_handler.stop_processor()
            
            # Land the drone if flying
            if hasattr(self, 'command_handler') and self.command_handler and self.command_handler.is_drone_flying():
                self.command_handler.emergency_land()
            
            # Stop video stream and release cameras
            if hasattr(self, 'camera_handler') and self.camera_handler:
                self.camera_handler.release()
            
            try:
                tello.streamoff()
                tello.end()
            except Exception as e:
                self.safe_print(f"Error during tello cleanup: {e}")
            
            self.destroy()
            sys.exit(0)

    def toggle_voice_recognition(self):
        """Toggle voice recognition on/off."""
        if self.voice_var.get():
            print("Enabling voice recognition...")
            if hasattr(self, 'speech_handler') and self.speech_handler:
                self.speech_handler.start_listening()
                self.log("Voice recognition enabled", "SPEECH")
        else:
            print("Disabling voice recognition...")
            if hasattr(self, 'speech_handler') and self.speech_handler:
                self.speech_handler.stop_listening()
                self.log("Voice recognition disabled", "SPEECH")

    def toggle_gesture_recognition(self):
        """Toggle gesture recognition on/off."""
        if self.gesture_var.get():
            print("Enabling gesture recognition...")
            if hasattr(self, 'command_handler') and self.command_handler:
                self.command_handler.set_modality("gesture")
                self.log("Gesture recognition enabled", "GESTURE")
        else:
            print("Disabling gesture recognition...")
            if hasattr(self, 'command_handler') and self.command_handler:
                self.command_handler.set_modality("idle")
                self.log("Gesture recognition disabled", "GESTURE")

    def toggle_camera_feed(self):
        """Toggle camera feed on/off."""
        if self.camera_var.get():
            print("Enabling camera feed...")
            if hasattr(self, 'camera_handler') and self.camera_handler:
                self.camera_handler.start_video()
                self.log("Camera feed enabled", "INFO")
        else:
            print("Disabling camera feed...")
            if hasattr(self, 'camera_handler') and self.camera_handler:
                self.camera_handler.stop_video()
                self.log("Camera feed disabled", "INFO")

    def toggle_drone_camera(self):
        """Toggle between drone and PC camera."""
        if self.drone_camera_var.get():
            print("Switching to drone camera...")
            if hasattr(self, 'camera_handler') and self.camera_handler:
                self.camera_handler.set_active_camera("drone")
                self.log("Switched to drone camera", "INFO")
        else:
            print("Switching to PC camera...")
            if hasattr(self, 'camera_handler') and self.camera_handler:
                self.camera_handler.set_active_camera("pc")
                self.log("Switched to PC camera", "INFO")

    def show_help(self):
        """Show help dialog."""
        help_text = """
        Tello Drone Control Help
        
        Keyboard Controls:
        - t: Takeoff
        - l: Land
        - Space: Hover
        - Arrow keys: Forward/Back/Left/Right
        - w/s: Up/Down
        - a/d: Rotate Left/Right
        - c: Toggle Camera
        - 1/2/3: Switch mode (Gesture/Audio/VLM)
        - 0: Idle mode
        
        Voice Commands:
        - "Take off"
        - "Land"
        - "Move forward/backward/left/right"
        - "Turn left/right"
        - "Move up/down"
        - "Hover"
        - "Flip forward/backward/left/right"
        
        Gestures (if model available):
        - Hand up: Move up
        - Hand down: Move down
        - Hand left/right: Move left/right
        - Open hand forward: Move forward
        - Open hand back: Move backward
        - Fist: Land
        """
        messagebox.showinfo("Help", help_text)

# Main entry point
def main():
    try:
        # Create app directly as root window (no separate root)
        app = DroneApp()
        app.mainloop()
    except Exception as e:
        # Use direct console output for critical errors to avoid recursion
        import sys
        sys.stderr.write(f"Critical error: {e}\n")
        try:
            # Emergency landing if app crashes
            tello.land()
            tello.streamoff()
        except Exception as landing_error:
            sys.stderr.write(f"Error during emergency landing: {landing_error}\n")

if __name__ == "__main__":
    main() 