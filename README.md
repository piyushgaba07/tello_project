

# Tello Drone Multimodal Control GUI

This Python application provides a comprehensive Graphical User Interface (GUI) built with Tkinter for controlling a DJI Tello drone. It offers multiple modes of interaction, including keyboard, hand gestures, voice commands (using Whisper), and visual analysis via a Vision Language Model (VLM).


## Features

*   **Graphical User Interface:** Easy-to-use interface built with Tkinter and ttk for modern styling.
*   **Live Video Feed:** Displays real-time video from either the Tello drone's camera or the local PC webcam.
*   **Multiple Control Modalities:**
    *   **Keyboard Control:** Fly the drone using standard keyboard keys (Arrows, WASD, QE, RF, Space, T, L).
    *   **Gesture Control:** Control the drone using hand gestures detected via MediaPipe and a custom-trained model (`gesture_model.h5`).
    *   **Voice Control:** Issue commands using your voice, transcribed by OpenAI's Whisper model.
    *   **Vision AI Analysis (VLM):** Ask questions in natural language about the content of the current video feed (using a model like LLaVA via `drone_control.py`).
*   **Camera Switching:** Seamlessly switch the video source between the drone and the PC webcam.
*   **Real-time Status:** Displays drone battery level, flight status (Landed/Flying), current command, and active control mode.
*   **Activity Log:** Provides a timestamped log of events, commands, errors, and detections, with color-coded levels for readability.
*   **VLM Interaction:** Dedicated panel to input text queries for the VLM and view its responses based on the camera feed.
*   **Responsive Controls:** Visual feedback (button highlighting) for active commands.
*   **Safe Shutdown:** Attempts to land the drone automatically when the application is closed.
*   **Cross-Platform:** Includes basic optimizations for different operating systems (e.g., macOS antialiasing).
*   **Modular:** Relies on a separate `drone_control.py` module for core drone communication, gesture model loading, and VLM processing logic.

## Prerequisites

*   **Python:** 3.8 or higher recommended.
*   **DJI Tello Drone:** The application is designed specifically for this drone.
*   **Network:** Your computer must be connected to the Tello drone's Wi-Fi network.
*   **Python Libraries:** Install the required libraries. You can typically install them using pip:
    ```bash
    pip install -r requirements.txt 
    ```
*   **Gesture Model:** A trained Keras model file (e.g., `gesture_model.h5`) is required for gesture control. Ensure this file is present in the same directory or that the path in `drone_control.py` is correct. The corresponding `gesture_labels.txt` (or similar, based on `gesture_labels` in `drone_control.py`) should also be present.
*   **Microphone:** Required for Voice Control mode. Ensure it's configured correctly on your system. You might need to adjust `AUDIO_DEVICE_INDEX` in `drone_control.py`.
*   **Webcam (Optional):** Required if you want to use the "Local PC" camera source for video feed or gesture control.
*   **VLM Backend (Optional):** The Vision AI Analysis feature requires a separate VLM setup (e.g., Ollama running a LLaVA model) configured within `drone_control.py`. The `process_vlm_input` function in that file handles the interaction.

## Installation & Setup

1.  **Clone this Repository:**

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place Gesture Model:** Download or ensure your trained `gesture_model.h5` and `gesture_labels.txt` (or equivalent based on `drone_control.py`) are in the same directory as the scripts, or update the path in `drone_control.py`.
4.  **Configure `drone_control.py`:**
    *   Verify Tello connection details (if needed, usually automatic with `djitellopy`).
    *   Set the correct `WHISPER_MODEL` name (e.g., "tiny", "base", "small"). Smaller models are faster but less accurate.
    *   Adjust `AUDIO_DEVICE_INDEX` if your microphone isn't the default. You can list devices using `speech_recognition.Microphone.list_microphone_names()`.
    *   Set `PHRASE_TIME_LIMIT` for voice commands if needed.
    *   Configure the `process_vlm_input` function to connect to your specific VLM backend (e.g., Ollama API endpoint and model name).
    *   Ensure gesture model loading path and label mapping are correct.
5.  **Connect to Tello:** Turn on your Tello drone and connect your computer to its Wi-Fi network.

## Usage

1.  **Run the Application:**
    ```bash
    python main_gui.py
    ```

2.  **Interface Overview:**
    *   **Left Pane:**
        *   **Camera Feed:** Shows the live video from the selected source (PC or Drone). Overlays show mode, camera, battery, and detected gesture (if applicable).
        *   **Vision AI Analysis:** Input field to type questions about the video feed, an "Ask AI" button, and a display area for the VLM's response.
        *   **Controls:** Buttons for Take Off/Land/Hover, Radio buttons to select Control Mode (Gesture, Voice, Vision Analysis, Idle) and Camera Source (Local PC, Tello Drone).
        *   **Directional Controls:** Buttons for manual flight (Up, Down, Left, Right, Forward, Backward, Rotate Left/Right, Flips). Keyboard shortcuts are also available.
    *   **Right Pane:**
        *   **Status:** Displays Battery Percentage, Flight Status, Current Command/Action, and Active Control Mode.
        *   **Activity Log:** Shows real-time logs from the application and drone interaction. Use the "Clear Log" button to clear the display.

3.  **Flying the Drone:**
    *   **Connect:** Ensure you are connected to the Tello Wi-Fi *before* starting the script.
    *   **Select Mode:** Choose your preferred control modality (Keyboard - implied when not Idle, Gesture, Voice). Idle mode disables direct flight commands but allows VLM queries.
    *   **Select Camera:** Choose the video source. Gesture and VLM analysis use the selected camera feed.
    *   **Take Off:** Press the "Take Off" button or 'T' key. The drone's status should change to "Flying".
    *   **Control:** Use the selected method (Keyboard, Gestures, Voice) to fly.
    *   **Vision Analysis:** Switch to "Vision Analysis" mode or simply type a query and click "Ask AI" (this might temporarily switch the mode internally for processing). The drone will hover while analyzing unless commanded otherwise by a different active mode beforehand.
    *   **Land:** Press the "Land" button or 'L' key.

## Control Modes Explained

*   **Idle:** No active drone control via gestures or voice. Keyboard/button commands (except VLM query) are ignored. Useful for observing or using VLM without accidental movement. The drone *will hover* if it was already airborne.
*   **Gesture:** Enables hand gesture recognition using the selected camera feed.
    *   Requires the gesture model (`.h5` file) to be loaded correctly.
    *   Hold a recognized gesture steadily for a short duration (approx. 5 frames) to trigger the corresponding command (e.g., "forward", "land", "flip").
    *   The specific gesture-to-command mapping is defined in `drone_control.py` and `execute_gesture_command` in the GUI script.
*   **Voice:** Activates the microphone to listen for voice commands.
    *   Uses the Whisper model defined in `drone_control.py`.
    *   Speak clearly after the "Listening for command..." log message appears.
    *   Recognized commands (e.g., "take off", "move forward", "turn left", "land") are executed. See `recognize_command` for recognized phrases.
*   **Vision Analysis (VLM):** Allows interaction with a Vision Language Model.
    *   Type your question about the current camera view into the "Vision AI Analysis" input box and click "Ask AI".
    *   The application sends the current frame and your query to the VLM (configured in `drone_control.py`).
    *   The AI's response is displayed in the designated text area.
    *   *This mode primarily analyzes the scene; it does not directly control the drone based on the VLM's response.*
    *   There's a short cooldown period between VLM queries.

## Keyboard Shortcuts

| Key         | Action                     |
| :---------- | :------------------------- |
| `↑` / `W`   | Move Forward               |
| `↓` / `S`   | Move Backward              |
| `←` / `A`   | Move Left                  |
| `→` / `D`   | Move Right                 |
| `Q`         | Rotate Counter-Clockwise   |
| `E`         | Rotate Clockwise           |
| `R`         | Move Up                    |
| `F`         | Move Down                  |
| `Space`     | Hover (Stop Movement)      |
| `T`         | Take Off                   |
| `L`         | Land                       |
| *(Flip keys may depend on implementation)* | *(e.g., numpad or specific letter keys if implemented)* |

*Note: Keyboard shortcuts are generally active when the mode is NOT Idle or VLM.*

## Troubleshooting & Notes

*   **Connection Issues:** Ensure you are connected to the Tello Wi-Fi *before* launching. Check firewall settings if connection fails.
*   **Camera Issues:** If the PC camera doesn't work, ensure no other application is using it and drivers are up-to-date. If the drone camera fails, try restarting the drone and the application.
*   **Microphone Issues:** Check system sound settings. Ensure the correct `AUDIO_DEVICE_INDEX` is set in `drone_control.py`.
*   **Gesture Model:** If gesture mode fails, verify the `.h5` model file exists, the path is correct in `drone_control.py`, and you have `tensorflow` installed.
*   **VLM Errors:** Ensure your VLM backend (e.g., Ollama) is running and accessible. Check the configuration in `process_vlm_input` within `drone_control.py`.
*   **Performance:** Whisper (especially larger models) and VLM processing can be resource-intensive. Performance may vary based on your hardware.
*   **Lag:** Network latency can affect responsiveness. Fly in an area with minimal Wi-Fi interference.
*   **Color Correction:** The code includes a function (`color_correct_drone_frame`) to improve the Tello camera's color accuracy.
*   **Emergency Landing:** The application attempts to land the drone if a critical error occurs or when closing the window, but always be prepared to manually stop the drone if necessary.

## License

*This project is licensed under the MIT License*
```
