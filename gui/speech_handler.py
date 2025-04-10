"""
Speech recognition module for the Tello Drone Control application.
"""

import speech_recognition as sr
import time
import queue
import threading
import builtins
from .whisper_processor import WhisperProcessor, CommandProcessor

# Original print for direct console output
_original_print = builtins.print

class SpeechHandler:
    """Handles speech recognition for voice commands."""
    def __init__(self, command_queue, device_index=None, model="tiny", phrase_time_limit=5):
        """Initialize the speech recognition handler.
        
        Note: During initialization, only console output should be used.
        The GUI logger will not be available until later."""
        self.command_queue = command_queue
        self.device_index = device_index
        self.model = model
        self.phrase_time_limit = phrase_time_limit
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.audio_processor = None
        self.listener_thread = None
        self.is_listening = False
        self.command_processor = CommandProcessor()
        
        # Initialize whisper if available
        try:
            self.audio_processor = WhisperProcessor(model=model)
            # Use direct console print during initialization to avoid GUI logging
            _original_print(f"Whisper initialized with model: {model}")
        except Exception as e:
            _original_print(f"Error initializing Whisper: {e}")
        
        # Print available microphones using direct console print
        self.print_available_mics()
        
        # Try to select the microphone
        try:
            self.microphone = sr.Microphone(device_index=device_index)
            _original_print(f"Microphone initialized with device index: {device_index}")
        except Exception as e:
            _original_print(f"Error initializing microphone: {e}")
    
    def safe_print(self, message):
        """Print safely regardless of GUI state."""
        # Always print to console using the original print
        _original_print(message)
    
    def print_available_mics(self):
        """List all available microphones using direct console print."""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            _original_print("Available microphones:")
            for i, mic in enumerate(mic_list):
                _original_print(f"  {i}: {mic}")
        except Exception as e:
            _original_print(f"Error listing microphones: {e}")
    
    def adjust_for_ambient_noise(self):
        """Adjust for ambient noise to improve recognition accuracy."""
        if not self.microphone:
            self.safe_print("No microphone available for ambient noise adjustment")
            return False
        
        try:
            self.safe_print("Adjusting for ambient noise... (please be quiet)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.safe_print(f"Adjusted energy threshold: {self.recognizer.energy_threshold}")
            return True
        except Exception as e:
            self.safe_print(f"Error adjusting for ambient noise: {e}")
            return False
    
    def start_listening(self):
        """Start the speech recognition listener in a separate thread."""
        if self.is_listening:
            self.safe_print("Already listening")
            return
        
        if not self.microphone:
            self.safe_print("No microphone available")
            return
        
        # First adjust for ambient noise
        if not self.adjust_for_ambient_noise():
            return
        
        # Start listening thread
        self.is_listening = True
        self.listener_thread = threading.Thread(target=self.listen_for_commands)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        self.safe_print("Speech recognition started")
    
    def stop_listening(self):
        """Stop the speech recognition listener."""
        self.is_listening = False
        if self.listener_thread:
            # Wait for thread to terminate
            self.listener_thread.join(timeout=1)
            self.listener_thread = None
        self.safe_print("Speech recognition stopped")
    
    def listen_for_commands(self):
        """Listen for voice commands and process them."""
        while self.is_listening:
            try:
                with self.microphone as source:
                    self.safe_print("Listening for commands...")
                    audio = self.recognizer.listen(source, phrase_time_limit=self.phrase_time_limit)
                    self.safe_print("Processing audio...")
                
                # Process with Whisper
                if self.audio_processor:
                    text = self.audio_processor.process_audio(audio)
                    if text:
                        self.safe_print(f"Recognized: {text}")
                        self.process_command(text)
                    else:
                        self.safe_print("Could not recognize audio")
                else:
                    self.safe_print("Whisper processor not available")
            
            except sr.WaitTimeoutError:
                self.safe_print("No speech detected within timeout")
            except Exception as e:
                self.safe_print(f"Error in speech recognition: {e}")
                time.sleep(1)  # Prevent rapid error loops
    
    def process_command(self, text):
        """Process a recognized command."""
        command = self.command_processor.process_text(text)
        if command:
            self.safe_print(f"Command detected: {command}")
            self.add_to_queue(command)
        else:
            self.safe_print("No command detected in text")
    
    def add_to_queue(self, command):
        """Add a command to the queue."""
        try:
            if not self.command_queue.full():
                self.command_queue.put(command)
                self.safe_print(f"Added command to queue: {command}")
            else:
                self.safe_print("Command queue is full")
        except Exception as e:
            self.safe_print(f"Error adding command to queue: {e}")
    
    def set_device_index(self, device_index):
        """Set the microphone device index."""
        self.device_index = device_index
        was_listening = self.is_listening
        
        # Stop listening if active
        if was_listening:
            self.stop_listening()
        
        # Reinitialize microphone
        try:
            self.microphone = sr.Microphone(device_index=device_index)
            self.safe_print(f"Microphone set to device index: {device_index}")
        except Exception as e:
            self.safe_print(f"Error setting microphone: {e}")
        
        # Resume listening if it was active
        if was_listening:
            self.start_listening()
    
    def set_model(self, model):
        """Set the Whisper model."""
        try:
            self.audio_processor = WhisperProcessor(model=model)
            self.model = model
            self.safe_print(f"Whisper model set to: {model}")
        except Exception as e:
            self.safe_print(f"Error setting Whisper model: {e}") 