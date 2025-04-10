"""
Gesture recognition and processing for the Tello Drone Control application.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

class GestureProcessor:
    """
    Processor for detecting and handling hand gestures for drone control.
    """
    def __init__(self, gesture_model=None, gesture_labels=None):
        """Initialize the gesture processor."""
        self.gesture_model = gesture_model
        self.gesture_labels = gesture_labels or {}
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture tracking variables
        self.pred_gesture = ""
        self.temp_gesture = ""
        self.gesture_count = 0
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # seconds between gestures
    
    def process_frame(self, frame, display_frame=None):
        """
        Process a frame for hand gestures.
        
        Args:
            frame: The original camera frame
            display_frame: Optional frame for visualization (if None, original frame is used)
            
        Returns:
            gesture: Detected gesture name or None
            processed_frame: Frame with hand landmarks drawn
        """
        if frame is None:
            return None, None
            
        # Use provided display frame or make a copy
        processed_frame = display_frame.copy() if display_frame is not None else frame.copy()
        
        # If no gesture model, just return the frame
        if self.gesture_model is None:
            return None, processed_frame
            
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)
        
        # No hands detected
        if not results.multi_hand_landmarks:
            self.temp_gesture = ""
            self.gesture_count = 0
            return None, processed_frame
            
        # Extract landmarks
        landmarks = []
        h, w, _ = frame.shape
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the display frame
            self.mp_draw.draw_landmarks(processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Extract x, y coordinates
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x * w)
                landmarks.append(lm.y * h)
                
        # Predict gesture
        try:
            # Reshape landmarks for model input
            landmarks_array = tf.expand_dims(landmarks, axis=0)
            
            # Get prediction from model
            predictions = self.gesture_model.predict(landmarks_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Check if confidence is high enough
            if confidence > 0.9:
                gesture_name = self.gesture_labels.get(predicted_class, "unknown")
                
                # Check if same gesture is sustained
                if gesture_name == self.temp_gesture:
                    self.gesture_count += 1
                else:
                    self.temp_gesture = gesture_name
                    self.gesture_count = 0
                
                # Number of consistent frames required for a gesture
                if self.gesture_count >= 5:
                    # Check cooldown
                    current_time = time.time()
                    if current_time - self.last_gesture_time > self.gesture_cooldown:
                        self.pred_gesture = gesture_name
                        self.gesture_count = 0
                        self.last_gesture_time = current_time
                        return gesture_name, processed_frame
            
        except Exception as e:
            print(f"Gesture prediction error: {e}")
        
        return None, processed_frame
    
    def get_last_predicted_gesture(self):
        """Get the last predicted gesture."""
        return self.pred_gesture
    
    def reset_prediction(self):
        """Reset the gesture prediction."""
        self.pred_gesture = ""
        self.temp_gesture = ""
        self.gesture_count = 0
    
    def set_cooldown(self, seconds):
        """Set the cooldown between gestures."""
        self.gesture_cooldown = max(0.1, seconds)  # Minimum 0.1 seconds
    
    def is_available(self):
        """Check if gesture recognition is available."""
        return self.gesture_model is not None 