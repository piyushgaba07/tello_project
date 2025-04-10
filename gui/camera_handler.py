"""
Camera handling module for the Tello Drone Control application.
"""

import cv2
import numpy as np
import time
import queue
from threading import Lock

class CameraHandler:
    """
    Handler for managing PC and drone camera feeds and processing.
    """
    def __init__(self, drone_video_address="udp://@0.0.0.0:11111", camera_idx=0, record_video=False, overlay_canvas=None):
        """Initialize the camera handler."""
        self.drone_video_address = drone_video_address
        self.pc_camera_index = camera_idx
        self.record_video = record_video
        self.overlay_canvas = overlay_canvas
        self.pc_cam = None
        self.active_camera = "pc"  # Default camera source
        self.is_video_active = True  # Video feed active flag
        self.frame_queue = queue.Queue(maxsize=1)
        self.lock = Lock()  # For thread safety
        
        # Try to initialize Tello from drone_control.py
        try:
            from .. import drone_control
            self.tello = drone_control.tello
            print("Tello drone initialized for camera handler")
        except Exception as e:
            print(f"Error initializing Tello drone reference: {e}")
            self.tello = None
        
        # Initialize PC camera
        self.initialize_pc_camera()
    
    def initialize_pc_camera(self):
        """Initialize the PC camera."""
        try:
            self.pc_cam = cv2.VideoCapture(self.pc_camera_index)
            if not self.pc_cam.isOpened():
                print(f"Warning: Could not open PC camera {self.pc_camera_index}")
                self.active_camera = "drone"  # Default to drone if PC camera fails
                return False
            return True
        except Exception as e:
            print(f"Error initializing PC camera: {e}")
            self.active_camera = "drone"
            return False
    
    def switch_camera(self):
        """Toggle between PC and drone camera."""
        with self.lock:
            if self.active_camera == "pc":
                print("Switching to drone camera")
                self.active_camera = "drone"
                return True
            else:
                print("Switching to PC camera")
                self.active_camera = "pc"
                # Verify PC camera is working
                if not self.pc_cam or not self.pc_cam.isOpened():
                    if not self.initialize_pc_camera():
                        print("PC camera not available, staying with drone camera")
                        self.active_camera = "drone"
                        return False
                return True
    
    def get_frame(self):
        """Get a frame from the active camera."""
        if not self.is_video_active:
            return None
            
        with self.lock:
            frame = None
            
            if self.active_camera == "pc":
                # Get PC camera frame
                if self.pc_cam and self.pc_cam.isOpened():
                    ret, frame = self.pc_cam.read()
                    if not ret:
                        print("Error reading PC camera frame, switching to drone camera")
                        self.active_camera = "drone"
                        # Try to get a drone frame instead
                        try:
                            if self.tello:
                                frame = self.tello.get_frame_read().frame
                                if frame is not None:
                                    frame = self.color_correct_drone_frame(frame)
                        except Exception as e:
                            print(f"Error reading drone camera after PC camera failed: {e}")
                            return None
                    else:
                        # Mirror PC camera for better UX
                        frame = cv2.flip(frame, 1)
                else:
                    # PC camera not available, switch to drone
                    self.active_camera = "drone"
                    try:
                        if self.tello:
                            frame = self.tello.get_frame_read().frame
                            if frame is not None:
                                frame = self.color_correct_drone_frame(frame)
                    except Exception as e:
                        print(f"Error reading drone camera: {e}")
                        return None
            else:
                # Get drone camera frame
                try:
                    if self.tello:
                        frame = self.tello.get_frame_read().frame
                        if frame is None:
                            print("Error getting drone camera frame, trying again")
                            time.sleep(0.1)
                            return None
                        # Apply color correction to drone feed
                        frame = self.color_correct_drone_frame(frame)
                except Exception as e:
                    print(f"Error reading drone camera: {e}")
                    return None
            
            # Update frame queue for VLM
            if frame is not None:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame.copy())
                    except:
                        pass
            
            return frame
    
    def color_correct_drone_frame(self, frame):
        """Apply color correction to drone camera feed to reduce blue tint."""
        if frame is None:
            return None
            
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split the LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Adjust the A and B channels to reduce blue tint
        a = cv2.add(a, np.ones_like(a) * 3, dtype=cv2.CV_8U)
        # Decrease B channel (blue-yellow) to reduce blue tint
        b = cv2.add(b, np.ones_like(b) * 15, dtype=cv2.CV_8U)  # Adding positive value shifts toward yellow
        
        # Merge the adjusted channels
        adjusted_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        corrected = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
        
        # Additional warming filter to counteract blue
        warming_layer = np.zeros_like(corrected, dtype=np.uint8)
        warming_layer[:] = [0, 30, 50]  # BGR warm tone
        corrected = cv2.addWeighted(corrected, 0.9, warming_layer, 0.1, 0)
        
        return corrected
    
    def release(self):
        """Release camera resources."""
        if self.pc_cam and self.pc_cam.isOpened():
            self.pc_cam.release()
        # Note: tello video stream is handled by the main application
    
    def get_active_camera(self):
        """Get the current active camera."""
        return self.active_camera
    
    def set_active_camera(self, camera):
        """Set the active camera.
        
        Args:
            camera (str): Either "pc" or "drone"
        """
        if camera in ["pc", "drone"]:
            self.active_camera = camera
            return True
        return False
    
    def start_video(self):
        """Start video feed."""
        self.is_video_active = True
        
    def stop_video(self):
        """Stop video feed."""
        self.is_video_active = False 