"""
Audio processing module for whisper-based speech recognition.
"""

import speech_recognition as sr
import time

class WhisperProcessor:
    """
    Handles audio processing using Whisper models for speech recognition.
    """
    def __init__(self, model="tiny"):
        """
        Initialize the WhisperProcessor.
        
        Args:
            model (str): The Whisper model to use. Options include:
                "tiny", "base", "small", "medium", "large"
        """
        self.model = model
        self.recognizer = sr.Recognizer()
        # Initialize energy threshold for better detection
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
    def process_audio(self, audio):
        """
        Process audio data using Whisper to get transcribed text.
        
        Args:
            audio: Audio data from speech_recognition
            
        Returns:
            str: Transcribed text or empty string on failure
        """
        try:
            # Return transcribed text using Whisper
            text = self.recognizer.recognize_whisper(audio, language="english", model=self.model)
            print(f"Whisper recognized: '{text}'")
            return text.strip()
        except sr.UnknownValueError:
            print("Whisper could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Whisper service; {e}")
            return ""
        except Exception as e:
            print(f"Error processing audio with Whisper: {e}")
            return ""


class CommandProcessor:
    """
    Processes transcribed text into drone commands.
    """
    def __init__(self):
        """Initialize the command processor with command keywords."""
        # Define command keywords and their variations
        self.command_keywords = {
            "takeoff": ["take off", "takeoff", "start", "begin flight", "take of", "takeoff", "start flying"],
            "land": ["land", "stop", "go down", "descend fully", "ground", "go landing", "land now"],
            "move forward": ["move forward", "go forward", "fly forward", "forward", "ahead", "straight", "front"],
            "move backward": ["move backward", "go back", "fly backward", "backward", "reverse", "back", "behind"],
            "move left": ["move left", "go left", "fly left", "left", "leftward", "to the left"],
            "move right": ["move right", "go right", "fly right", "right", "rightward", "to the right"],
            "move up": ["move up", "go up", "fly up", "up", "upward", "higher", "ascend", "rise"],
            "move down": ["move down", "go down", "fly down", "down", "downward", "lower", "descend", "sink"],
            "turn left": ["turn left", "rotate left", "spin left", "twist left"],
            "turn right": ["turn right", "rotate right", "spin right", "twist right"],
            "hover": ["hover", "stay", "pause", "hold position", "stop moving", "freeze", "hold", "wait"],
            "flip forward": ["flip forward", "do a front flip", "front flip", "forward flip"],
            "flip backward": ["flip backward", "do a back flip", "back flip", "backward flip"],
            "flip left": ["flip left", "do a left flip", "left flip"],
            "flip right": ["flip right", "do a right flip", "right flip"],
        }
        
    def process_text(self, text):
        """
        Process text to identify drone commands.
        
        Args:
            text (str): Text to process for commands
            
        Returns:
            str: Recognized command or None if no command detected
        """
        if not text:
            return None
            
        text = text.lower().strip()
        
        # First check for exact matches
        for command, variations in self.command_keywords.items():
            if text in variations:
                return command
                
        # Then check for partial matches (if a variation is contained in the text)
        for command, variations in self.command_keywords.items():
            for variation in variations:
                if variation in text:
                    return command
                    
        # No command detected
        print(f"No command detected in: '{text}'")
        return None 