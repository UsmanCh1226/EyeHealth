import cv2
import json
import time
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# Local module imports
from constants import (
    EAR_THRESHOLD, MAR_THRESHOLD, DIST_THRESHOLD_CM,
    LOGGING_INTERVAL_SEC, LOG_FILE_PATH, CALIBRATION_FILE_PATH,
    FONT, TEXT_COLOR_GREEN, TEXT_COLOR_RED, TEXT_COLOR_YELLOW,
    ALERT_BG_COLOR, NORMAL_BG_COLOR,
    L_START, L_END, R_START, R_END, MOUTH_TOP, MOUTH_BOTTOM,
    KNOWN_FACE_WIDTH_CM
)
from utils import (
    eye_aspect_ratio, mouth_aspect_ratio,
    calculate_distance, draw_text_with_background
)
from logger import EyeHealthLogger

# --- Global State Variables ---
BLINK_COUNTER = 0
TOTAL_BLINKS = 0
YAWN_COUNTER = 0
TOTAL_YAWNS = 0
IS_DROWSY = False
IS_TOO_CLOSE = False

# Load Focal Length from Calibration
try:
    with open(CALIBRATION_FILE_PATH, 'r') as f:
        CALIBRATION_DATA = json.load(f)
        FOCAL_LENGTH = CALIBRATION_DATA.get('FocalLength', 0.0)
    print(f"Loaded Focal Length: {FOCAL_LENGTH}")
except FileNotFoundError:
    print(f"Calibration file not found at {CALIBRATION_FILE_PATH}. Please run calibration.py first.")
    # Exit here if FOCAL_LENGTH is absolutely necessary
    # For robust production code, we might set FOCAL_LENGTH = 1.0 and prompt calibration.
    # For now, we'll exit as intended by the previous logic.
    exit()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Initialize Camera and Logger
cap = cv2.VideoCapture(0)
health_logger = EyeHealthLogger(LOG_FILE_PATH, LOGGING_INTERVAL_SEC)
start_time = time.time()
last_log_time = start_time

def get_landmarks(image):
    """Processes the image and returns a list of normalized landmarks."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        h, w, _ = image.shape
        # Scale normalized landmarks (0-1) to pixel coordinates (0-W, 0-H)
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
        return landmarks, image
    return None, image

def main_loop():
    global BLINK_COUNTER, TOTAL_BLINKS, YAWN_COUNTER, TOTAL_YAWNS, IS_DROWSY, IS_TOO_CLOSE
    global last_log_time, start_time

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        landmarks, image = get_landmarks(image)
        current_time = time.time()
        
        # Initialize distance in case no face is detected
        current_distance_cm = 0.0

        if landmarks:
            # --- 1. Eye Aspect Ratio (EAR) for Drowsiness/Blinking ---
            
            # Get the indices of the points needed for the calculation (simplified 6 points)
            # Left Eye: 362, 385, 380, 373, 386, 263
            l_eye_coords = np.array([landmarks[i] for i in [362, 385, 380, 373, 386, 263]])
            # Right Eye: 133, 160, 158, 153, 144, 33
            r_eye_coords = np.array([landmarks[i] for i in [133, 160, 158, 153, 144, 33]])

            # Calculate EARs
            left_ear = eye_aspect_ratio(l_eye_coords)
            right_ear = eye_aspect_ratio(r_eye_coords)
            avg_ear = (left_ear + right_ear) / 2.0

            # Blink Detection Logic
            if avg_ear < EAR_THRESHOLD:
                BLINK_COUNTER += 1
            else:
                if BLINK_COUNTER > 1: # Require more than 1 frame closure to count as a blink
                    TOTAL_BLINKS += 1
                BLINK_COUNTER = 0

            # Drowsiness check (sustained low EAR)
            # 30 frames is roughly 1 second at 30 FPS
            if BLINK_COUNTER > 30: 
                IS_DROWSY = True
            else:
                IS_DROWSY = False


            # --- 2. Mouth Aspect Ratio (MAR) for Yawning ---
            mouth_coords = np.array([landmarks[i] for i in [MOUTH_TOP, MOUTH_BOTTOM, 61, 291]]) # Indices: 13 (top), 14 (bottom), 61 (left), 291 (right)
            mar = mouth_aspect_ratio(mouth_coords)

            if mar > MAR_THRESHOLD:
                YAWN_COUNTER += 1
            else:
                # Only register a yawn if mouth was open for a sustained period
                if YAWN_COUNTER > 10: 
                    TOTAL_YAWNS += 1
                YAWN_COUNTER = 0


            # --- 3. Distance Calculation (Proximity Alert) ---
            # Indices 234 and 454 define the rough horizontal width of the face/head
            face_pixel_width = dist.euclidean(landmarks[234], landmarks[454])
            current_distance_cm = calculate_distance(FOCAL_LENGTH, KNOWN_FACE_WIDTH_CM, face_pixel_width)

            if current_distance_cm < DIST_THRESHOLD_CM:
                IS_TOO_CLOSE = True
            else:
                IS_TOO_CLOSE = False

            # --- 4. Drawing Metrics & Alerts ---
            
            # Distance Display
            distance_text = f"Distance: {current_distance_cm:.1f} cm"
            if IS_TOO_CLOSE:
                image = draw_text_with_background(image, distance_text, (20, 50), FONT, 0.7, TEXT_COLOR_RED, ALERT_BG_COLOR)
                alert_text = f"TOO CLOSE! Maintain distance > {DIST_THRESHOLD_CM:.0f} cm."
                image = draw_text_with_background(image, alert_text, (20, 90), FONT, 0.9, TEXT_COLOR_RED, ALERT_BG_COLOR, thickness=2)
            else:
                image = draw_text_with_background(image, distance_text, (20, 50), FONT, 0.7, TEXT_COLOR_GREEN, NORMAL_BG_COLOR)

            # Eye and Yawn Metrics Display
            blink_text = f"Total Blinks: {TOTAL_BLINKS}"
            yawn_text = f"Total Yawns: {TOTAL_YAWNS}"
            ear_text = f"EAR: {avg_ear:.2f}"

            image = draw_text_with_background(image, blink_text, (20, 130), FONT, 0.6, TEXT_COLOR_WHITE, NORMAL_BG_COLOR)
            image = draw_text_with_background(image, yawn_text, (20, 160), FONT, 0.6, TEXT_COLOR_WHITE, NORMAL_BG_COLOR)
            image = draw_text_with_background(image, ear_text, (20, 190), FONT, 0.6, TEXT_COLOR_WHITE, NORMAL_BG_COLOR)

            # Drowsiness Alert
            if IS_DROWSY:
                drowsy_alert = "DROWSINESS DETECTED! Rest eyes."
                image = draw_text_with_background(image, drowsy_alert, (20, 230), FONT, 0.9, TEXT_COLOR_YELLOW, ALERT_BG_COLOR, thickness=2)

            # Draw landmarks for visualization
            # Note: R_START and L_START are just start indices, using the calculated eye/mouth points is more precise
            for point in l_eye_coords:
                cv2.circle(image, tuple(point.astype(int)), 1, (0, 255, 0), -1)
            for point in r_eye_coords:
                cv2.circle(image, tuple(point.astype(int)), 1, (0, 255, 0), -1)
            for point in mouth_coords:
                cv2.circle(image, tuple(point.astype(int)), 1, (0, 255, 255), -1)


        # --- 5. Logging Logic ---
        if current_time - last_log_time >= LOGGING_INTERVAL_SEC:
            # Calculate Blinks Per Minute (BPM)
            duration_minutes = (current_time - start_time) / 60.0
            
            # Handle potential division by zero if start_time was just reset
            total_blinks_bpm = TOTAL_BLINKS / duration_minutes if duration_minutes > 0 else 0 

            # Log the data
            health_logger.log_metrics(
                total_blinks_bpm=total_blinks_bpm,
                current_distance_cm=current_distance_cm,
                is_drowsy=IS_DROWSY,
                is_too_close=IS_TOO_CLOSE,
                yawn_count=TOTAL_YAWNS
            )

            # Reset counts for the next logging interval
            TOTAL_BLINKS = 0
            TOTAL_YAWNS = 0
            start_time = current_time # Start fresh timing for the next calculation
            last_log_time = current_time

        cv2.imshow('Eye Health Monitor', image)

        # Break loop with 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_loop()