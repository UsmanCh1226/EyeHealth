import cv2
import dlib
import numpy as np
import time
import os
import sys

# --- Configuration ---
# Path to the dlib facial landmark predictor model
MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"

# Use os.path functions to create a robust, absolute path to the model file.
# This ensures it works regardless of the current working directory.
try:
    # Get the directory where the current script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the shape predictor file
    SHAPE_PREDICTOR_PATH = os.path.join(SCRIPT_DIR, MODEL_FILENAME)
except NameError:
    # Fallback if __file__ is not defined (e.g., in an interactive shell)
    SHAPE_PREDICTOR_PATH = MODEL_FILENAME
    
# Constants for Eye Aspect Ratio (EAR)
# Threshold for eye closure
EYE_AR_THRESH = 0.25
# Number of consecutive frames the EAR must be below the threshold for a blink to register
EYE_AR_CONSEC_FRAMES = 3

# Constants for Drowsiness/Alert
# Max consecutive frames where eyes are closed to trigger alert (~3 seconds at 30 FPS)
DROWSINESS_THRESHOLD = 50 
TIME_FOR_REST_ALERT = 120 # Suggest a break after this many seconds (2 minutes)

# --- Global State Variables ---
COUNTER = 0
TOTAL_BLINKS = 0
START_TIME = time.time()
LAST_REST_TIME = START_TIME
ALERT_ACTIVE = False

# --- Helper Functions ---

def euclidean_distance(ptA, ptB):
    """Calculates the Euclidean distance between two 2D points."""
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye.
    """
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal eye landmarks
    C = euclidean_distance(eye[0], eye[3])

    # Compute the Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

def draw_text_box(frame, text, position, font_scale=0.7, color=(0, 255, 0), thickness=2):
    """Draws text with a background box for better visibility."""
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    
    # Calculate box coordinates
    box_start = (x - 5, y - text_h - baseline - 5)
    box_end = (x + text_w + 5, y + baseline + 5)
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, box_start, box_end, (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw the text
    cv2.putText(frame, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return frame

# --- Main Logic ---

def main():
    global COUNTER, TOTAL_BLINKS, ALERT_ACTIVE, LAST_REST_TIME

    # --- PATH FINAL CHECK ---
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        # If the robust path still fails, print the absolute path it tried to use.
        print(f"\n!!! FATAL ERROR: Dlib model file not found.")
        print(f"The script looked for the file at the ABSOLUTE path:")
        print(f"--> {SHAPE_PREDICTOR_PATH}")
        print("Please ensure the file is present at this location.")
        return 
    # --- END PATH CHECK ---


    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    try:
        # Load the predictor using the robust path
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    except RuntimeError as e:
        print(f"Runtime Error during Dlib loading: {e}")
        return

    # Indices for the left and right eye points (from dlib's 68-point model)
    (lStart, lEnd) = (42, 48) # Left eye points
    (rStart, rEnd) = (36, 42) # Right eye points

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream (webcam).")
        return

    frames = 0
    fps_start_time = time.time()
    
    print("Eye Health Monitor started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames += 1
        current_time = time.time()

        # Calculate FPS
        if (current_time - fps_start_time) > 1:
            fps = frames / (current_time - fps_start_time)
            fps_start_time = current_time
            frames = 0
        else:
            try:
                fps = frames / (current_time - fps_start_time)
            except ZeroDivisionError:
                fps = 0

        # Convert the frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        ALERT_ACTIVE = False
        status_text = "STATUS: Monitoring"
        status_color = (0, 255, 0) # Green

        if len(rects) > 0:
            rect = rects[0]
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # --- Blink and Drowsiness Detection Logic ---
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= DROWSINESS_THRESHOLD:
                    ALERT_ACTIVE = True
                    status_text = "!!! DROWSINESS ALERT !!!"
                    status_color = (0, 0, 255) # Red
                    
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES and COUNTER < DROWSINESS_THRESHOLD:
                    TOTAL_BLINKS += 1
                COUNTER = 0

            # Calculate Blink Rate (BPM)
            elapsed_time = current_time - START_TIME
            blink_rate = int((TOTAL_BLINKS / elapsed_time) * 60) if elapsed_time > 0 else 0

            # 20-20-20 Rule Reminder
            time_since_last_rest = current_time - LAST_REST_TIME
            rest_suggestion_text = ""
            if time_since_last_rest > TIME_FOR_REST_ALERT:
                rest_suggestion_text = f"20-20-20 Rule: Break time! ({int(time_since_last_rest)}s)"
            
            # --- Display Information ---
            
            # 1. Main Status Alert
            frame = draw_text_box(frame, status_text, (50, 30), font_scale=0.9, color=status_color, thickness=2)

            # 2. Key Metrics
            frame = draw_text_box(frame, f"EAR: {ear:.2f}", (10, frame.shape[0] - 80), color=(255, 255, 0))
            frame = draw_text_box(frame, f"Blinks: {TOTAL_BLINKS}", (10, frame.shape[0] - 40), color=(255, 255, 0))
            frame = draw_text_box(frame, f"BPM: {blink_rate}", (150, frame.shape[0] - 40), color=(255, 255, 0))

            # 3. Rest Suggestion
            if rest_suggestion_text:
                 frame = draw_text_box(frame, rest_suggestion_text, (frame.shape[1] - 350, frame.shape[0] - 40), color=(0, 165, 255))
                 
            # 4. FPS
            frame = draw_text_box(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 80, 30), font_scale=0.6, color=(100, 100, 255))
        
        else:
             status_text = "STATUS: No Face Detected"
             frame = draw_text_box(frame, status_text, (50, 30), font_scale=0.9, color=(0, 165, 255), thickness=2)

        cv2.imshow("Eye Health Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()