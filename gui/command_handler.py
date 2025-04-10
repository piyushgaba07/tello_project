"""
Command handling module for the Tello Drone Control application.
"""

import queue
import time
from threading import Thread, Event
import sys
import os

# Add parent directory to path so we can import drone_control
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CommandHandler:
    """
    Handler for processing and executing drone commands.
    """
    def __init__(self, command_queue=None):
        """Initialize the command handler."""
        # Try to initialize Tello from drone_control.py
        try:
            from .. import drone_control
            self.tello = drone_control.tello
            print("Tello drone initialized for command handler")
        except Exception as e:
            print(f"Error initializing Tello drone reference: {e}")
            self.tello = None
            
        self.command_queue = command_queue if command_queue else queue.Queue(maxsize=5)
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Command states
        self.drone_in_air = False
        self.active_modality = "idle"  # Default to idle mode
        self.stop_event = Event()
        self.processor_thread = None
        
        # VLM settings
        self.last_vlm_command_time = 0
        self.vlm_cooldown = 3  # seconds
        
        # Callbacks
        self.on_command_executed = None
        self.on_vlm_response = None
    
    def set_modality(self, modality):
        """Set the active control modality."""
        self.active_modality = modality
        
        # Stop movement when changing modes
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            print(f"Error stopping movement: {e}")
    
    def start_processor(self):
        """Start the command processor thread."""
        if self.processor_thread is None or not self.processor_thread.is_alive():
            self.stop_event.clear()
            self.processor_thread = Thread(target=self.command_processor, 
                                         name="CommandProcessor", daemon=True)
            self.processor_thread.start()
    
    def stop_processor(self):
        """Stop the command processor thread."""
        self.stop_event.set()
        if self.processor_thread:
            self.processor_thread.join(timeout=2)
    
    def command_processor(self):
        """Process commands from the queue based on active modality."""
        print("Command processor started")
        
        while not self.stop_event.is_set():
            try:
                # Get a command from the queue with a timeout
                command_input = self.command_queue.get(timeout=1)
                
                # Process based on active modality
                if self.active_modality == "audio":
                    self.process_audio_command(command_input)
                elif self.active_modality == "vlm":
                    self.process_vlm_command(command_input)
                # Note: gesture commands are handled directly in the video processing loop
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue  # No command in queue
            except Exception as e:
                print(f"Command processing error: {e}")
                # Ensure task_done is called even on error if item was retrieved
                if 'command_input' in locals():
                    try: 
                        self.command_queue.task_done()
                    except ValueError: 
                        pass  # Already marked done
        
        print("Command processor stopped")
    
    def process_audio_command(self, command_text):
        """Process a command from speech recognition."""
        print(f"Processing voice command: '{command_text}'")
        
        # Recognize the command
        drone_command = self.recognize_command(command_text)
        if drone_command not in ["Unknown Command", "Already Landed", "Already Flying"]:
            self.execute_command(drone_command)
        else:
            print(f"Voice command ignored: {drone_command}")
    
    def process_vlm_command(self, command_text):
        """Process a command from VLM without controlling the drone."""
        from drone_control import process_vlm_input  # Import here to avoid circular imports
        
        current_time = time.time()
        if current_time - self.last_vlm_command_time > self.vlm_cooldown:
            print(f"Processing VLM query: '{command_text}'")
            
            # Get the latest frame for processing
            frame = None
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                print("No frame available for VLM processing")
                return
                
            self.last_vlm_command_time = current_time
            
            # Process the VLM command with the current frame
            vlm_response = process_vlm_input(command_text, frame)
            
            # Call callback with response if available
            if self.on_vlm_response:
                self.on_vlm_response(command_text, vlm_response)
                
            # Also try to extract a drone command from VLM response
            drone_command = self.recognize_command(vlm_response)
            if drone_command not in ["Unknown Command", "Already Landed", "Already Flying"]:
                self.execute_command(drone_command)
                
        else:
            print(f"VLM query '{command_text}' skipped due to cooldown")
    
    def process_gesture_command(self, gesture_name):
        """Process a gesture command directly."""
        if not gesture_name:
            return
            
        print(f"Processing gesture: {gesture_name}")
        
        # Map gestures to drone commands
        gesture_mapping = {
            "forward": "move_forward",
            "backward": "move_back",
            "left": "move_left",
            "right": "move_right",
            "up": "move_up",
            "down": "move_down",
            "flip": "flip_f",
            "land": "land"
        }
        
        # Execute the command if mapped
        if gesture_name in gesture_mapping:
            self.execute_command(gesture_mapping[gesture_name])
    
    def recognize_command(self, text):
        """Check if recognized text matches any drone command and convert to API command."""
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

        print(f"No matching command found for: '{text}'")
        return "Unknown Command"
    
    def execute_command(self, command):
        """Execute a drone command with feedback."""
        # Don't execute commands in idle mode
        if self.active_modality == "idle":
            print(f"Command ignored (Idle mode): {command}")
            return False
            
        try:
            print(f"Executing command: {command}")
            
            # Execute the command based on type
            if command == "takeoff":
                if not self.drone_in_air:
                    self.tello.takeoff()
                    self.drone_in_air = True
                else:
                    print("Drone already in air")
                    return False
                    
            elif command == "land":
                if self.drone_in_air:
                    self.tello.land()
                    self.drone_in_air = False
                else:
                    print("Drone already on ground")
                    return False
                    
            elif command == "hover":
                self.tello.send_rc_control(0, 0, 0, 0)
                    
            elif self.drone_in_air:
                # Movement commands
                if command == "move_forward":
                    self.tello.move_forward(30)
                elif command == "move_back":
                    self.tello.move_back(30)
                elif command == "move_left":
                    self.tello.move_left(30)
                elif command == "move_right":
                    self.tello.move_right(30)
                elif command == "move_up":
                    self.tello.move_up(30)
                elif command == "move_down":
                    self.tello.move_down(30)
                elif command == "rotate_counter_clockwise":
                    self.tello.rotate_counter_clockwise(45)
                elif command == "rotate_clockwise":
                    self.tello.rotate_clockwise(45)
                    
                # Direct RC commands (for gesture control)
                elif command == "rc_forward":
                    self.tello.send_rc_control(0, 20, 0, 0)
                elif command == "rc_back":
                    self.tello.send_rc_control(0, -20, 0, 0)
                elif command == "rc_left":
                    self.tello.send_rc_control(-20, 0, 0, 0)
                elif command == "rc_right":
                    self.tello.send_rc_control(20, 0, 0, 0)
                elif command == "rc_up":
                    self.tello.send_rc_control(0, 0, 20, 0)
                elif command == "rc_down":
                    self.tello.send_rc_control(0, 0, -20, 0)
                    
                # Flip commands
                elif command == "flip_f":
                    self.tello.flip("f")
                elif command == "flip_b":
                    self.tello.flip("b")
                elif command == "flip_l":
                    self.tello.flip("l")
                elif command == "flip_r":
                    self.tello.flip("r")
                else:
                    print(f"Unknown command: {command}")
                    return False
            else:
                print(f"Cannot execute '{command}' while landed. Try 'takeoff'.")
                return False
            
            # Call callback if provided
            if self.on_command_executed:
                self.on_command_executed(command)
                
            return True
            
        except Exception as e:
            print(f"Error executing command '{command}': {e}")
            
            # Attempt to stabilize after error
            if self.drone_in_air:
                try:
                    print("Attempting to hover after error")
                    self.tello.send_rc_control(0, 0, 0, 0)
                except Exception as hover_e:
                    print(f"Could not stabilize after error: {hover_e}")
            
            return False
    
    def enqueue_command(self, command_text):
        """Add a command to the queue."""
        if not self.command_queue.full():
            self.command_queue.put(command_text)
            return True
        else:
            print("Command queue full, discarding older command")
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.command_queue.put_nowait(command_text)
                return True
            except queue.Full:
                print("Failed to add command to queue")
                return False
    
    def emergency_land(self):
        """Emergency land the drone."""
        try:
            print("EMERGENCY LANDING")
            if self.drone_in_air:
                self.tello.land()
                self.drone_in_air = False
            
            # Ensure connection ends cleanly
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            print(f"Error during emergency landing: {e}")
    
    def is_drone_flying(self):
        """Check if drone is currently in the air."""
        return self.drone_in_air 