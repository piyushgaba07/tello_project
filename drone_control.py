import queue
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import speech_recognition as sr # Use speech_recognition for Whisper
from ollama import Client # For VLM
import base64 # For VLM image encoding
from djitellopy import Tello
import threading
import os

# --- Configuration ---
CAMERA_INDEX = 0
WHISPER_MODEL = "base.en"  # Faster Whisper model (options: tiny.en, base.en, small.en, medium.en, large)
AUDIO_DEVICE_INDEX = 0 # Use default microphone
PHRASE_TIME_LIMIT = 4 # Seconds to listen for a phrase - Increased
VLM_TIMEOUT = 120 # Seconds to wait for VLM response - Increased from 60 to 120

# --- Initialize Components ---
# Tello Drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
# Start drone video stream immediately
tello.streamon()
print("Drone video stream started")

# List available microphones for debugging
try:
    print("\nAvailable Microphones:")
    mic_names = sr.Microphone.list_microphone_names()
    for index, name in enumerate(mic_names):
        print(f"  Index {index}: {name}")
    print("\nUsing default microphone (or specify AUDIO_DEVICE_INDEX).\n")
except Exception as mic_e:
    print(f"Could not list microphones: {mic_e}\n")

# Gesture Recognition Model
try:
    gesture_model = tf.keras.models.load_model("gesture-model.h5", compile=False)
    # Gesture Labels
    gesture_labels = {0: "forward", 1: "backward", 2: "up", 3: "down", 4: "left", 5: "right", 6: "flip", 7: "land"}
    print("Gesture model loaded successfully.")
except Exception as e:
    print(f"Error loading gesture model: {e}. Gesture mode will be unavailable.")
    gesture_model = None
    gesture_labels = {}

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

# Ollama Client (for VLM)
ollama_client = Client(timeout=VLM_TIMEOUT)

# --- Global Variables & Queues ---
active_modality = "gesture" if gesture_model else "audio" # Default mode (fallback to audio if gesture model failed)
active_camera = "pc"  # Default camera source ("pc" or "drone")
command_queue = queue.Queue(maxsize=5) # Queue for voice/text commands
frame_queue = queue.Queue(maxsize=1) # Queue for the latest frame (for VLM)
stop_event = threading.Event()
drone_in_air = False # Track takeoff state
last_ping_time = time.time() # For tracking when we last communicated with the drone
PING_INTERVAL = 2  # seconds between pings to keep connection (reduced from 5)

# --- Drone Command Definitions ---
COMMANDS = {
    "takeoff": ["take off", "start", "begin flight", "arise", "launch"],
    "land": ["land", "stop", "descend fully", "ground"],
    "move forward": ["move forward", "go forward", "fly forward"],
    "move backward": ["move backward", "go back", "fly backward"],
    "turn left": ["turn left", "rotate left"],
    "turn right": ["turn right", "rotate right"],
    "move left": ["move left", "go left", "fly left", "strafe left"], # Added strafe
    "move right": ["move right", "go right", "fly right", "strafe right"], # Added strafe
    "move up": ["move up", "ascend", "fly up"],
    "move down": ["move down", "descend", "fly down"],
    "hover": ["hover", "stay", "hold position", "stop moving"],
    "flip forward": ["flip forward", "do a front flip"],
    "flip backward": ["flip backward", "do a back flip"],
    "flip left": ["flip left", "do a left flip"],
    "flip right": ["flip right", "do a right flip"],
}

# --- Command Recognition and Execution ---
def recognize_command(text):
    """Check if recognized text matches any drone command."""
    global drone_in_air
    text = text.lower().strip()
    if not text:
        return "Unknown Command"

    # Check exact commands first
    for action, phrases in COMMANDS.items():
        if text in phrases:
             # Prevent landing/takeoff commands if already in that state (simple check)
            if action == "land" and not drone_in_air: return "Already Landed"
            if action == "takeoff" and drone_in_air: return "Already Flying"
            return action

    # Check partial matches (be careful with these)
    for action, phrases in COMMANDS.items():
        if any(phrase in text for phrase in phrases):
            if action == "land" and not drone_in_air: return "Already Landed"
            if action == "takeoff" and drone_in_air: return "Already Flying"
            return action

    print(f"No matching command found for: '{text}'")
    return "Unknown Command"

def execute_command(command):
    """Send commands to the Tello drone."""
    global drone_in_air
    try:
        print(f"Executing command: {command}")
        if command == "takeoff":
            if not drone_in_air:
                print("🚀 Taking off...")
                tello.takeoff()
                drone_in_air = True
            else: print("Drone already in air.")
        elif command == "land":
            if drone_in_air:
                print("🛬 Landing...")
                tello.land()
                drone_in_air = False
            else: print("Drone already on ground.")
        elif drone_in_air: # Most commands only work if flying
            if command == "move forward":
                tello.move_forward(30)
            elif command == "move backward":
                tello.move_back(30)
            elif command == "turn left":
                tello.rotate_counter_clockwise(45)
            elif command == "turn right":
                tello.rotate_clockwise(45)
            elif command == "move left": # Added
                tello.move_left(30)
            elif command == "move right": # Added
                tello.move_right(30)
            elif command == "move up":
                tello.move_up(30) # Increased distance
            elif command == "move down":
                tello.move_down(30) # Increased distance
            elif command == "hover":
                print("⏸️ Hovering...")
                tello.send_rc_control(0, 0, 0, 0) # Explicit hover
            elif command == "flip forward":
                tello.flip("f")
            elif command == "flip backward":
                tello.flip("b")
            elif command == "flip left":
                tello.flip("l")
            elif command == "flip right":
                tello.flip("r")
            else:
                print(f"❌ Command '{command}' known but not executable in current state (or unhandled).")
        elif command not in ["takeoff", "land"]:
             print(f"Cannot execute '{command}' while landed. Try 'takeoff'.")

    except Exception as e:
        print(f"Error executing command '{command}': {e}")
        # Attempt to stabilize after error if in air
        if drone_in_air:
            try:
                print("Attempting to hover after error...")
                tello.send_rc_control(0, 0, 0, 0)
            except Exception as hover_e:
                print(f"Could not stabilize after error: {hover_e}")

# --- Input Handling Threads ---
def listen_for_voice_commands(cmd_q, stop_ev):
    """Listens for voice commands using Whisper and puts them into the queue."""
    print(f"Starting voice listener (Whisper model: {WHISPER_MODEL})...")
    r = sr.Recognizer()
    # Don't create mic here anymore
    # mic = sr.Microphone(device_index=AUDIO_DEVICE_INDEX)

    # Initial ambient noise adjustment (outside the loop)
    # We need a temporary mic instance for this
    try:
        with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
            print("Adjusting for ambient noise (initial)...")
            r.adjust_for_ambient_noise(source, duration=2) # Increased duration
            print(f"Adjusted energy threshold: {r.energy_threshold}") # Added logging
            print("Ready for voice commands.")
    except Exception as adjust_e:
        print(f"Error during initial ambient noise adjustment: {adjust_e}")
        print("Voice recognition might be unreliable.")
        # Set a default energy threshold if adjustment failed
        r.energy_threshold = 300 # A common default
        print(f"Using default energy threshold: {r.energy_threshold}")


    while not stop_ev.is_set():
        # DIAGNOSTIC: Check if loop is running and modality
        # print(f"Voice listener loop running. Active mode: {active_modality}") # Can be noisy

        if active_modality not in ["audio", "vlm"]: # Only listen actively if needed
            time.sleep(0.5)
            continue

        # DIAGNOSTIC: Confirm attempt to listen
        print("Attempting to listen...")
        try:
            # Re-initialize Microphone instance inside the loop
            print(f"Creating Microphone instance for device {AUDIO_DEVICE_INDEX}...")
            with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                 # Optional: Dynamic energy adjustment (can be slow)
                 # print("Adjusting dynamic energy...")
                 # r.adjust_for_ambient_noise(source, duration=0.2)

                 # DIAGNOSTIC: Add print before listen call within context manager
                 print("Calling r.listen...")
                 audio = r.listen(source, phrase_time_limit=PHRASE_TIME_LIMIT)
                 # DIAGNOSTIC: Confirm listen finished
                 print("r.listen() finished.")

            print("Processing audio...")
            # Use the chosen Whisper model
            command_text = r.recognize_whisper(audio, language="english", model=WHISPER_MODEL)
            command_text = command_text.strip()
            print(f"Whisper recognized: '{command_text}'")
            if command_text:
                if not cmd_q.full():
                    cmd_q.put(command_text)
                else:
                    print("Command queue full, discarding older command.")
                    try: cmd_q.get_nowait()
                    except queue.Empty: pass
                    try: cmd_q.put_nowait(command_text)
                    except queue.Full: pass # Should not happen

        except sr.WaitTimeoutError:
            print("No command heard.")
        except sr.UnknownValueError:
            print("Whisper could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Whisper service; {e}")
        except Exception as e:
            # Print specific error when creating Microphone instance fails
            if "Creating Microphone instance" in str(e):
                 print(f"Error creating Microphone instance: {e}")
            else:
                 print(f"Voice listening error: {e}")
            time.sleep(1)

    print("Voice listener stopped.")

def read_text_input(cmd_q, stop_ev):
    """Reads text input from the console (primarily for VLM mode)."""
    print("Starting text input reader. Press Enter after typing command.")
    while not stop_ev.is_set():
        try:
            if active_modality == "vlm": # Only prompt if VLM mode is active
                command = input("Enter VLM command (or press Ctrl+C to exit): ")
                if command and not stop_ev.is_set():
                    print(f"Text command received: {command}")
                    if not cmd_q.full():
                        cmd_q.put(command)
                    else:
                        print("Command queue full, discarding older command.")
                        try: cmd_q.get_nowait()
                        except queue.Empty: pass
                        try: cmd_q.put_nowait(command)
                        except queue.Full: pass
                elif stop_ev.is_set():
                    break
            else:
                time.sleep(0.5) # Don't block if not in VLM mode

        except EOFError:
            print("Text input stream closed.")
            break
        except KeyboardInterrupt:
            print("\nCtrl+C detected in text input.")
            if not stop_ev.is_set(): stop_ev.set()
            break
        except Exception as e:
            if not stop_ev.is_set():
                 print(f"Text input error: {e}")
                 time.sleep(1)
            break

    print("Text input reader stopped.")

# --- VLM Processing ---
def process_vlm_input(command_text, frame):
    """Sends command and frame to LLaVA, returns response."""
    if frame is None:
        print("VLM Error: No frame available.")
        return "Error: Missing frame."
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        print(f"Sending to LLaVA: Command='{command_text}', Image size: {len(frame_base64)} bytes (Timeout: {VLM_TIMEOUT}s)")
        response = ollama_client.chat(
            model='llava',
            messages=[
                {
                    'role': 'user',
                    'content': command_text,
                    'images': [frame_base64]
                }
            ]
        )
        vlm_response = response['message']['content']
        print(f"LLaVA Output: {vlm_response}")
        return vlm_response

    except Exception as e:
        print(f"Error interacting with Ollama/LLaVA: {e}")
        return f"Error: {e}"
    
# Add this function after the other function definitions, before the main loop:

def color_correct_drone_frame(frame):
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
    # Make sure to specify the output data type
    a = cv2.add(a, np.ones_like(a) * 3, dtype=cv2.CV_8U)
    # Decrease B channel (blue-yellow) to reduce blue tint
    b = cv2.add(b, np.ones_like(b) * 15, dtype=cv2.CV_8U)  # Adding positive value shifts toward yellow
    
    # Merge the adjusted channels
    adjusted_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    corrected = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    
    # Additional warming filter to counteract blue - fix type issue
    warming_layer = np.zeros_like(corrected, dtype=np.uint8)
    warming_layer[:] = [0, 30, 50]  # BGR warm tone
    corrected = cv2.addWeighted(corrected, 0.9, warming_layer, 0.1, 0)
    
    return corrected


# --- Central Command Processor Thread ---
def command_processor(cmd_q, frame_q, stop_ev):
    """Processes commands from the queue based on the active modality."""
    print("Starting command processor...")
    last_vlm_command_time = 0
    vlm_cooldown = 3 # Cooldown between VLM commands

    while not stop_ev.is_set():
        try:
            command_input = cmd_q.get(timeout=1) # Wait 1 sec for a command
            current_time = time.time()

            if active_modality == "audio":
                print(f"Audio Mode: Processing '{command_input}'")
                drone_command = recognize_command(command_input)
                if drone_command not in ["Unknown Command", "Already Landed", "Already Flying"]:
                    execute_command(drone_command)
                else:
                    print(f"Audio command ignored: {drone_command}")

            elif active_modality == "vlm":
                 if current_time - last_vlm_command_time > vlm_cooldown:
                    print(f"VLM Mode: Processing '{command_input}'")
                    frame = None
                    try:
                        frame = frame_q.get_nowait() # Get latest frame
                    except queue.Empty:
                        print("VLM Warning: No frame available for processing.")
                        # Optionally skip or proceed without frame? For now, skip.
                        cmd_q.task_done()
                        continue

                    last_vlm_command_time = current_time
                    vlm_response = process_vlm_input(command_input, frame)

                    # --- Basic VLM Response Interpretation ---
                    # Attempt to extract a drone command from the VLM's text response
                    # This is VERY basic and likely needs significant improvement
                    drone_command = recognize_command(vlm_response)
                    print(f"VLM Response interpreted as: {drone_command}")
                    if drone_command not in ["Unknown Command", "Already Landed", "Already Flying"]:
                         execute_command(drone_command)
                    else:
                         print(f"VLM command ignored: {drone_command}")
                    # --- End Basic Interpretation ---
                 else:
                      print(f"VLM command '{command_input}' skipped due to cooldown.")


            # Gesture mode commands are handled directly in the main loop

            cmd_q.task_done() # Mark command as processed

        except queue.Empty:
            continue # No command in queue, loop again
        except Exception as e:
            print(f"Command processing error: {e}")
            # Ensure task_done is called even on error if item was retrieved
            if 'command_input' in locals():
                 try: cmd_q.task_done()
                 except ValueError: pass # Already marked done
            time.sleep(1)

    print("Command processor stopped.")


# --- Main Loop (Video, Gesture, Key Input) ---
def run_drone_interface(frame_q, stop_ev):
    """Main loop for video feed, gesture detection, and mode switching."""
    global active_modality, active_camera
    
    # Initialize PC camera
    pc_cam = cv2.VideoCapture(CAMERA_INDEX)
    if not pc_cam.isOpened():
        print(f"Error: Could not open PC camera {CAMERA_INDEX}")
        print("Continuing with drone camera only.")
        active_camera = "drone"  # Default to drone camera if PC camera fails
    
    pred = ""
    temp = ""
    count = 0
    print(f"Starting main interface loop. Initial mode: {active_modality}")
    print("Press 'a' for Audio, 'g' for Gesture, 'v' for VLM, 'c' to toggle Camera Source, 'q' to Quit.")

    while not stop_ev.is_set():
        # Get frame based on active camera source
        if active_camera == "pc":
            ret, frame = pc_cam.read()
            if not ret:
                print("Error reading PC camera frame, switching to drone camera.")
                active_camera = "drone"
                continue
            frame = cv2.flip(frame, 1)  # Mirror PC camera for better UX
        else:  # active_camera == "drone"
            try:
                frame = tello.get_frame_read().frame
                if frame is None:
                    print("Error getting drone camera frame, trying again.")
                    time.sleep(0.1)
                    continue
                # Apply color correction to drone feed
                frame = color_correct_drone_frame(frame)
            except Exception as e:
                print(f"Error reading drone camera: {e}, trying again.")
                time.sleep(0.1)
                continue
        
        display_frame = frame.copy()  # Use a copy for drawing
        y, x, _ = frame.shape

        # Keep latest frame in queue for VLM
        if not frame_q.full():
            frame_q.put(frame.copy())
        else:
            try: 
                frame_q.get_nowait()
                frame_q.put_nowait(frame.copy())
            except queue.Empty: pass
            except queue.Full: pass

        # --- Gesture Detection (Only if mode is active and model loaded) ---
        if active_modality == "gesture" and gesture_model:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb_img)
            landmarks = []

            if res.multi_hand_landmarks:
                for handslms in res.multi_hand_landmarks:
                    # Draw landmarks on the display frame
                    draw.draw_landmarks(display_frame, handslms, mpHands.HAND_CONNECTIONS)
                    for lm in handslms.landmark:
                        landmarks.append(lm.x * x)
                        landmarks.append(lm.y * y)

                # Predict Gesture
                try:
                    prob = gesture_model.predict(tf.expand_dims(landmarks, axis=0), verbose=0) # Reduce verbosity
                    className = np.argmax(prob)

                    if prob[0][className] > 0.9: # Confidence threshold
                        gesture = gesture_labels.get(className, "unknown")

                        if gesture == temp:
                            count += 1
                        else:
                            temp = gesture
                            count = 0

                        if count == 5: # Require 5 consecutive frames
                            pred = temp
                            count = 0
                            print(f"Gesture detected: {pred}")
                            # Execute gesture command directly (no queue)
                            if pred == "forward": tello.send_rc_control(0, 20, 0, 0)
                            elif pred == "backward": tello.send_rc_control(0, -20, 0, 0)
                            elif pred == "right": tello.send_rc_control(20, 0, 0, 0)
                            elif pred == "left": tello.send_rc_control(-20, 0, 0, 0)
                            elif pred == "up": tello.send_rc_control(0, 0, 20, 0)
                            elif pred == "down": tello.send_rc_control(0, 0, -20, 0)
                            elif pred == "flip": execute_command("flip forward") # Example flip
                            elif pred == "land": execute_command("land")
                            # Add a small delay to prevent rapid-fire commands
                            time.sleep(0.5)
                            # Reset RC control after gesture action
                            tello.send_rc_control(0, 0, 0, 0)


                except Exception as gesture_e:
                    print(f"Gesture prediction error: {gesture_e}")
            else:
                 # No hands detected, reset RC control if gesture mode is active
                 tello.send_rc_control(0, 0, 0, 0)


        # --- Display Info ---
        cv2.putText(display_frame, f"Mode: {active_modality.upper()} ('a'/'g'/'v')", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Camera: {active_camera.upper()} ('c')", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Battery: {tello.get_battery()}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if active_modality == "gesture":
             cv2.putText(display_frame, f"Gesture: {pred}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drone Control Interface", display_frame)

        # --- Key Handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit key pressed.")
            stop_event.set()
            break
        elif key == ord("a"):
            if active_modality != "audio":
                print("Switching to AUDIO mode.")
                active_modality = "audio"
                tello.send_rc_control(0, 0, 0, 0) # Stop movement on mode switch
        elif key == ord("g"):
             if gesture_model: # Only switch if model loaded
                if active_modality != "gesture":
                    print("Switching to GESTURE mode.")
                    active_modality = "gesture"
                    tello.send_rc_control(0, 0, 0, 0) # Stop movement
             else:
                  print("Gesture model not available.")
        elif key == ord("v"):
            if active_modality != "vlm":
                print("Switching to VLM mode.")
                active_modality = "vlm"
                tello.send_rc_control(0, 0, 0, 0) # Stop movement
        elif key == ord("c"):
            # Toggle between PC and drone camera
            if active_camera == "pc":
                print("Switching to DRONE camera.")
                active_camera = "drone"
            else:
                print("Switching to PC camera.")
                active_camera = "pc"
                # Verify PC camera is still working
                if not pc_cam.isOpened():
                    try:
                        pc_cam = cv2.VideoCapture(CAMERA_INDEX)
                        if not pc_cam.isOpened():
                            print("PC camera not available, staying with DRONE camera.")
                            active_camera = "drone"
                    except Exception as cam_e:
                        print(f"Error opening PC camera: {cam_e}")
                        active_camera = "drone"

    # --- Cleanup ---
    if pc_cam.isOpened():
        pc_cam.release()
    cv2.destroyAllWindows()
    print("Main interface loop stopped.")

# Process detected gesture (or none) for each frame
def process_gesture(gesture_name):
    global active_modality
    
    if active_modality != "gesture":
        return # Only process gestures when in gesture mode
        
    process_command(gesture_name, source="gesture")

# Keep the drone connection alive by sending regular pings
def keep_connection_alive():
    global last_ping_time, tello
    
    while not stop_event.is_set():
        current_time = time.time()
        if current_time - last_ping_time > PING_INTERVAL:
            try:
                # Send a battery status request as a keepalive ping
                tello.get_battery()
                last_ping_time = current_time
                print("Ping sent to drone")
            except Exception as e:
                print(f"Error sending ping to drone: {e}")
        
        # Sleep to avoid using too much CPU
        time.sleep(1)

def keep_drone_alive(app):
    """Keeps the drone connection alive by sending periodic commands.
    
    This function runs in a separate thread and sends periodic pings to the drone
    to prevent automatic disconnection due to inactivity. It also monitors battery
    levels and performs emergency landing if levels are critically low.
    
    Args:
        app: The DroneApp instance to access drone and logging
    """
    MIN_BATTERY = 15  # Minimum battery percentage before forced landing
    CRITICAL_BATTERY = 10  # Critical battery level for emergency measures
    CHECK_INTERVAL = 5  # Check interval in seconds
    
    app.safe_print("Starting keep-alive monitoring thread")
    
    while app.keep_alive_active:
        try:
            # Only proceed if drone is connected
            if app.drone and app.drone.is_connected:
                # Get battery level
                try:
                    battery = app.drone.get_battery()
                    
                    # If we have a GUI, update the battery display
                    if hasattr(app, 'log_display'):
                        app.log_message(f"Battery: {battery}%")
                    else:
                        app.safe_print(f"Battery: {battery}%")
                    
                    # Handle low battery conditions
                    if battery <= CRITICAL_BATTERY:
                        app.safe_print("CRITICAL BATTERY LEVEL! Emergency landing initiated.")
                        if app.drone.is_flying:
                            app.drone.land()
                            app.safe_print("Emergency landing completed")
                    elif battery <= MIN_BATTERY and app.drone.is_flying:
                        app.safe_print("LOW BATTERY! Automatic landing initiated.")
                        app.drone.land()
                        app.safe_print("Automatic landing completed")
                        
                except Exception as e:
                    app.safe_print(f"Error checking battery: {e}")
                
                # Send periodic command to keep connection alive (if not actively flying)
                if not app.drone.is_flying:
                    app.drone.send_command("command")  # Heartbeat command
            
        except Exception as e:
            app.safe_print(f"Error in keep-alive thread: {e}")
        
        # Sleep for the check interval
        time.sleep(CHECK_INTERVAL)
    
    app.safe_print("Keep-alive monitoring thread stopped")

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing Drone Controller...")

    # Create threads
    threads = []
    threads.append(threading.Thread(target=listen_for_voice_commands, args=(command_queue, stop_event), name="VoiceListener", daemon=True))
    threads.append(threading.Thread(target=read_text_input, args=(command_queue, stop_event), name="TextInput", daemon=True))
    threads.append(threading.Thread(target=command_processor, args=(command_queue, frame_queue, stop_event), name="CommandProcessor", daemon=True))
    threads.append(threading.Thread(target=keep_connection_alive, name="KeepAlive", daemon=True))

    try:
        # Start background threads
        for t in threads:
            t.start()

        # Run main interface loop in the main thread
        run_drone_interface(frame_queue, stop_event)

    except KeyboardInterrupt:
        print("\nCtrl+C detected in main execution. Shutting down...")
        if not stop_event.is_set():
            stop_event.set()
    except Exception as e:
        print(f"\nAn unexpected error occurred in main execution: {e}")
        if not stop_event.is_set():
            stop_event.set()
    finally:
        # --- Graceful Shutdown ---
        print("Initiating shutdown sequence...")
        if not stop_event.is_set():
            stop_event.set() # Ensure stop event is set

        print("Waiting for threads to finish...")
        # Wait for daemon threads implicitly or explicitly if needed (daemons might exit abruptly)
        # time.sleep(2) # Give threads a moment

        print("Landing drone...")
        try:
            # Ensure drone lands if it was flying
            if drone_in_air:
                tello.land()
            else:
                 # If already landed, just ensure connection ends cleanly
                 tello.send_rc_control(0,0,0,0) # Send zero command before ending
        except Exception as land_e:
            print(f"Error during landing/shutdown: {land_e}")
        finally:
             try:
                  # Stop the video stream before ending
                  tello.streamoff()
                  tello.end()
             except Exception as end_e:
                  print(f"Error during tello cleanup: {end_e}")

        print("Shutdown complete.")