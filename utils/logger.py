"""
Logger module for the Tello Drone Control application.
"""

import time
import builtins

# Store reference to original print to avoid recursion
_original_print = builtins.print

class Logger:
    """
    Logger class to handle application logging with multiple levels and GUI integration.
    """
    def __init__(self, log_queue=None, max_logs=100):
        """Initialize the logger with a queue for GUI display."""
        self.log_queue = log_queue
        self.max_logs = max_logs
        self.gui_instance = None
        self.gui_ready = False
    
    def set_gui_instance(self, gui_instance):
        """Set the GUI instance for direct logging."""
        self.gui_instance = gui_instance
        # Check if GUI has log_display attribute
        self.gui_ready = hasattr(gui_instance, 'log_display')
    
    def log(self, message, level="INFO"):
        """Log a message with specified level."""
        # Print to console using the original print to avoid recursion
        _original_print(f"[{level}] {message}")
        
        # Add to queue if available
        if self.log_queue:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = {
                "timestamp": timestamp,
                "message": message,
                "level": level
            }
            
            if self.log_queue.full():
                try:
                    self.log_queue.get_nowait()  # Remove oldest log
                except:
                    pass
            
            try:
                self.log_queue.put_nowait(log_entry)
            except:
                pass
        
        # Log directly to GUI if available and ready
        if self.gui_ready and self.gui_instance and hasattr(self.gui_instance, 'log_display'):
            try:
                # Using direct call to log_display to avoid recursion
                self.gui_instance.log_display.add_log(message, level)
            except Exception:
                # GUI not fully initialized yet or other error
                pass
    
    def clear(self):
        """Clear the log queue."""
        if self.log_queue:
            while not self.log_queue.empty():
                try:
                    self.log_queue.get_nowait()
                except:
                    break

# Create a global logger instance
logger = Logger()

# Custom print function to redirect to logger
def custom_print(*args, **kwargs):
    """Replacement for built-in print that logs to the GUI."""
    # Call original print for console output
    _original_print(*args, **kwargs)
    
    # Log to the global logger
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
    
    # Only add to queue if available
    if hasattr(logger, 'log_queue') and logger.log_queue:
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = {
                "timestamp": timestamp,
                "message": message,
                "level": level
            }
            
            if logger.log_queue.full():
                try:
                    logger.log_queue.get_nowait()  # Remove oldest log
                except:
                    pass
            logger.log_queue.put_nowait(log_entry)
        except Exception:
            pass  # Silently ignore queue errors
    
    # Only try to log to GUI if the gui is ready and all components exist
    try:
        if (logger.gui_ready and logger.gui_instance and 
            hasattr(logger.gui_instance, 'log_display') and 
            logger.gui_instance.log_display is not None):
            logger.gui_instance.log_display.add_log(message, level)
    except Exception:
        pass  # Silently ignore GUI errors 