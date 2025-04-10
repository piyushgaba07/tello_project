"""
UI Components for the Tello Drone Control application.
"""

import tkinter as tk
from tkinter import ttk, font, scrolledtext
import time

class StyledFrame(ttk.Frame):
    """A styled frame with consistent appearance."""
    def __init__(self, parent, **kwargs):
        style = kwargs.pop('style', 'TFrame')
        super().__init__(parent, style=style, **kwargs)

class HeaderLabel(ttk.Label):
    """A styled header label."""
    def __init__(self, parent, text, **kwargs):
        style = kwargs.pop('style', 'Header.TLabel')
        super().__init__(parent, text=text, style=style, **kwargs)

class ControlButton(ttk.Button):
    """A styled button for drone controls."""
    def __init__(self, parent, text, command, **kwargs):
        style = kwargs.pop('style', 'Control.TButton')
        super().__init__(parent, text=text, command=command, style=style, **kwargs)

class LogDisplay(scrolledtext.ScrolledText):
    """A text widget for displaying logs with colorful formatting."""
    def __init__(self, parent, height=10, width=50, **kwargs):
        font_obj = kwargs.pop('font', None)
        super().__init__(parent, height=height, width=width, wrap=tk.WORD, **kwargs)
        self.config(state=tk.DISABLED)  # Make read-only
        
        # Configure tags for different log levels
        from utils.config import LOG_LEVELS
        for level, color in LOG_LEVELS.items():
            self.tag_configure(level, foreground=color)
    
    def add_log(self, message, level="INFO"):
        """Add a log entry with timestamp and appropriate color."""
        self.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.insert(tk.END, f"[{timestamp}] ", "TIMESTAMP")
        self.insert(tk.END, f"{message}\n", level)
        self.see(tk.END)  # Auto-scroll to bottom
        self.config(state=tk.DISABLED)
    
    def clear(self):
        """Clear all logs."""
        self.config(state=tk.NORMAL)
        self.delete(1.0, tk.END)
        self.config(state=tk.DISABLED)

class StatusBar(ttk.Frame):
    """A status bar to display important information."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Default variables
        self.battery_var = tk.StringVar(value="-- %")
        self.flight_status_var = tk.StringVar(value="Landed")
        self.mode_var = tk.StringVar(value="IDLE")
        self.command_var = tk.StringVar(value="None")
        
        # Layout
        self.create_layout()
    
    def create_layout(self):
        """Create the layout for the status bar."""
        # Battery status
        battery_frame = ttk.Frame(self)
        battery_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(battery_frame, text="Battery:").pack(side=tk.LEFT, padx=5)
        ttk.Label(battery_frame, textvariable=self.battery_var).pack(side=tk.LEFT)
        
        # Flight status
        flight_frame = ttk.Frame(self)
        flight_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(flight_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        ttk.Label(flight_frame, textvariable=self.flight_status_var).pack(side=tk.LEFT)
        
        # Mode
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Label(mode_frame, textvariable=self.mode_var).pack(side=tk.LEFT)
        
        # Current command
        cmd_frame = ttk.Frame(self)
        cmd_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(cmd_frame, text="Command:").pack(side=tk.LEFT, padx=5)
        ttk.Label(cmd_frame, textvariable=self.command_var).pack(side=tk.LEFT)
    
    def update_battery(self, value):
        """Update the battery level."""
        self.battery_var.set(f"{value}%")
    
    def update_flight_status(self, status):
        """Update the flight status."""
        self.flight_status_var.set(status)
    
    def update_mode(self, mode):
        """Update the control mode."""
        self.mode_var.set(mode.upper())
    
    def update_command(self, command):
        """Update the current command."""
        self.command_var.set(command)

class ControlPanel(ttk.LabelFrame):
    """A control panel with directional controls."""
    def __init__(self, parent, command_callback, **kwargs):
        super().__init__(parent, **kwargs)
        self.command_callback = command_callback
        self.active_buttons = {}
        self.create_layout()
    
    def create_layout(self):
        """Create the directional control layout."""
        # Create grid for directional buttons
        dir_grid = ttk.Frame(self)
        dir_grid.pack(padx=10, pady=10)
        
        # Up button (row 0, column 1)
        up_btn = ttk.Button(dir_grid, text="Up ↑", command=lambda: self.command_callback("move_up"))
        up_btn.grid(row=0, column=1, padx=5, pady=5)
        self.active_buttons["move_up"] = up_btn
        
        # Left button (row 1, column 0)
        left_btn = ttk.Button(dir_grid, text="← Left", command=lambda: self.command_callback("move_left"))
        left_btn.grid(row=1, column=0, padx=5, pady=5)
        self.active_buttons["move_left"] = left_btn
        
        # Forward button (row 1, column 1)
        forward_btn = ttk.Button(dir_grid, text="Forward", command=lambda: self.command_callback("move_forward"))
        forward_btn.grid(row=1, column=1, padx=5, pady=5)
        self.active_buttons["move_forward"] = forward_btn
        
        # Right button (row 1, column 2)
        right_btn = ttk.Button(dir_grid, text="Right →", command=lambda: self.command_callback("move_right"))
        right_btn.grid(row=1, column=2, padx=5, pady=5)
        self.active_buttons["move_right"] = right_btn
        
        # Down button (row 2, column 1)
        down_btn = ttk.Button(dir_grid, text="Down ↓", command=lambda: self.command_callback("move_down"))
        down_btn.grid(row=2, column=1, padx=5, pady=5)
        self.active_buttons["move_down"] = down_btn
        
        # Add rotation and backward buttons
        rot_frame = ttk.Frame(self)
        rot_frame.pack(padx=10, pady=(0, 10))
        
        rot_left_btn = ttk.Button(rot_frame, text="Rotate Left", 
                                 command=lambda: self.command_callback("rotate_counter_clockwise"))
        rot_left_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["rotate_counter_clockwise"] = rot_left_btn
        
        backward_btn = ttk.Button(rot_frame, text="Backward", 
                                 command=lambda: self.command_callback("move_back"))
        backward_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["move_back"] = backward_btn
        
        rot_right_btn = ttk.Button(rot_frame, text="Rotate Right", 
                                  command=lambda: self.command_callback("rotate_clockwise"))
        rot_right_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["rotate_clockwise"] = rot_right_btn
        
        # Add hover button to the frame
        hover_btn = ttk.Button(rot_frame, text="Hover", 
                             command=lambda: self.command_callback("hover"))
        hover_btn.pack(side=tk.LEFT, padx=5)
        self.active_buttons["hover"] = hover_btn
    
    def highlight_button(self, command):
        """Highlight the active button."""
        if command in self.active_buttons:
            self.active_buttons[command].state(['pressed'])
    
    def reset_button(self, command):
        """Reset button highlight."""
        if command in self.active_buttons:
            self.active_buttons[command].state(['!pressed'])


class VlmPanel(ttk.LabelFrame):
    """A panel for VLM interaction."""
    def __init__(self, parent, send_callback, **kwargs):
        super().__init__(parent, **kwargs)
        self.send_callback = send_callback
        self.create_layout()
    
    def create_layout(self):
        """Create the VLM panel layout."""
        # Informational label
        info_label = ttk.Label(self, text="Ask the AI about what it sees in the camera feed:", 
                             wraplength=300, foreground="#0066cc")
        info_label.pack(fill=tk.X, padx=5, pady=(10, 10))
        
        # Input frame
        input_frame = ttk.Frame(self)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Entry field
        self.entry = ttk.Entry(input_frame)
        self.entry.pack(fill=tk.X, expand=True, padx=5, pady=8, ipady=4)
        
        # Send button
        send_btn = ttk.Button(input_frame, text="Ask AI", command=self.on_send, 
                            style="VLM.TButton", width=15)
        send_btn.pack(padx=8, pady=8)
        
        # Response area
        response_frame = ttk.Frame(self, style="VLM.TFrame")
        response_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Response header
        header_label = ttk.Label(response_frame, text="AI Response", 
                               style="VLMHeader.TLabel", anchor="center")
        header_label.pack(fill=tk.X, padx=0, pady=0)
        
        # Response text widget
        text_container = ttk.Frame(response_frame)
        text_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.response_text = scrolledtext.ScrolledText(text_container, wrap=tk.WORD, 
                                                     height=6, width=40, 
                                                     background="#f9f9ff", foreground="#333333",
                                                     borderwidth=1, relief="sunken")
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.response_text.config(state=tk.DISABLED)
    
    def on_send(self):
        """Handle the send button click."""
        query = self.entry.get().strip()
        if query:
            self.send_callback(query)
            self.entry.delete(0, tk.END)  # Clear entry
    
    def update_response(self, query, response):
        """Update the response text widget."""
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        
        # Format the response with timestamp
        timestamp = time.strftime("%H:%M:%S")
        formatted_response = f"[{timestamp}] Query: {query}\n\n{response}"
        
        self.response_text.insert(tk.END, formatted_response)
        
        # Add tags for styling different parts
        self.response_text.tag_configure("timestamp", foreground="#0066cc", 
                                       font=font.Font(family="Helvetica", size=9, weight="bold"))
        self.response_text.tag_configure("query", foreground="#009933", 
                                       font=font.Font(family="Helvetica", size=9, slant="italic"))
        self.response_text.tag_configure("response", foreground="#333333")
        
        # Apply tags
        self.response_text.tag_add("timestamp", "1.0", "1.10")
        self.response_text.tag_add("query", "1.11", "2.0")
        self.response_text.tag_add("response", "3.0", "end")
        
        self.response_text.config(state=tk.DISABLED)
        
    def clear_response(self):
        """Clear the response text."""
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.config(state=tk.DISABLED) 