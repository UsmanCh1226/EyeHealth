import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import math

# --- Configuration and Constants ---

# Standard camera settings for distance calculation (needs to be calibrated)
# FOCAL_LENGTH is the distance of the user's face from the camera (in the units you measured for calibration)
# This value is read from the calibration file, but we use a default if the file is missing.
DEFAULT_FOCAL_LENGTH = 1720.45 

# Target distance in mm (e.g., a healthy viewing distance)
TARGET_DISTANCE_MM = 500  

# Thresholds in mm (e.g., alert if closer than 400mm or farther than 700mm)
TOO_CLOSE_MM = 400
TOO_FAR_MM = 700

# File paths
CALIBRATION_FILE = "calibration.txt"
LOG_FILE = "eye_monitoring_log.csv"

# Time interval for logging data (in seconds)
LOG_INTERVAL_SEC = 5

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Landmark Indices for Face Mesh (Based on MediaPipe documentation) ---
# We use specific landmarks to measure the distance in 3D space
# Standard points for a robust 3D head distance measurement:
# 152: Nose tip (front)
# 454: Right ear/cheek side
# 234: Left ear/cheek side
# 199: Chin/Mouth center
# 1: Forehead/Top center

# Indices for the eyes (used for drawing/status, not primary distance calc)
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246]
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 390, 249, 263, 373, 374]

# Indices for key points used in 3D distance calculation (from the front of the face)
FRONT_FACE_LANDMARKS = [10, 152, 234, 454] 


# --- Helper Functions ---

def load_calibration():
    """Loads the focal length from a calibration file."""
    focal_length = DEFAULT_FOCAL_LENGTH
    try:
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, 'r') as f:
                line = f.readline().strip()
                focal_length = float(line)
        print(f"Calibration loaded: Focal Length = {focal_length:.2f}")
    except Exception as e:
        print(f"Warning: Could not load calibration file. Using default. Error: {e}")
    return focal_length

def initialize_log():
    """Initializes the CSV log file with headers."""
    # Check if file exists to avoid writing headers multiple times
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Timestamp', 
                'Distance_MM', 
                'Status', 
                'Recommended_Distance_MM',
                'Raw_3D_Face_Width_Pixels'
            ])
    print(f"Log file initialized: {LOG_FILE}")

def log_data(distance_mm, status, face_width_px):
    """Writes the current data to the CSV log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp, 
            f"{distance_mm:.2f}", 
            status, 
            TARGET_DISTANCE_MM, 
            f"{face_width_px:.2f}"
        ])

def calculate_distance(landmarks, frame_w, focal_length):
    """
    Calculates the distance from the camera based on face width in pixels (P)
    using the perspective formula: D = (W * F) / P
    
    W (Actual Width) is the known actual distance between the two points in mm.
    F (Focal Length) is the calibrated focal length in pixels.
    P (Pixel Width) is the measured pixel distance between the two points on the frame.
    """
    
    # We use the distance between the two side cheek landmarks (234 and 454) as a reference.
    # Typical adult face width (W) at this reference point is approximately 140 mm.
    # This value is a standard estimate and should be validated through calibration.
    KNOWN_FACE_WIDTH_MM = 140.0

    # Get the coordinates of the two key side points
    p1 = landmarks[234] 
    p2 = landmarks[454]

    # Calculate the pixel distance (P) between these two points
    # First convert normalized coordinates (0 to 1) to actual pixel coordinates
    p1_px = (int(p1.x * frame_w), int(p1.y * frame_w))
    p2_px = (int(p2.x * frame_w), int(p2.y * frame_w))

    # Calculate the Euclidean distance in pixels
    pixel_width = math.sqrt((p2_px[0] - p1_px[0])**2 + (p2_px[1] - p1_px[1])**2)
    
    if pixel_width > 0:
        # Distance (D) in mm = (KNOWN_FACE_WIDTH_MM * FOCAL_LENGTH) / pixel_width
        distance_mm = (KNOWN_FACE_WIDTH_MM * focal_length) / pixel_width
    else:
        distance_mm = 0
        
    return distance_mm, pixel_width

# --- Main Monitoring Function ---

def start_monitoring():
    """Initializes and runs the eye monitoring application."""
    
    # Load settings and initialize logging
    focal_length = load_calibration()
    initialize_log()
    
    # Timing for log interval
    last_log_time = time.time()

    # Video Capture Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Monitoring started. Press 'q' to exit.")

    # MediaPipe Face Mesh Pipeline
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Ignoring empty camera frame.")
                break

            # Flip the image horizontally for a natural selfie-view display
            frame = cv2.flip(frame, 1) 
            
            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # CRITICAL FIX LOCATION: Define h and w *inside* the loop
            h, w, _ = frame.shape
            
            # Process the frame
            results = face_mesh.process(rgb_frame)

            # Initialize variables for the loop iteration
            distance_mm = 0.0
            face_width_px = 0.0
            status_text = "Searching for Face..."
            status_color = (0, 0, 255) # Red (Default: Searching)
            
            # --- Processing Landmarks ---
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # Calculate distance
                    distance_mm, face_width_px = calculate_distance(
                        face_landmarks.landmark, 
                        w, 
                        focal_length
                    )
                    
                    # Determine Status
                    if distance_mm < TOO_CLOSE_MM and distance_mm > 0:
                        status_text = "TOO CLOSE! (Distance: {:.0f}mm)".format(distance_mm)
                        status_color = (0, 0, 255)  # Red
                    elif distance_mm > TOO_FAR_MM:
                        status_text = "TOO FAR! (Distance: {:.0f}mm)".format(distance_mm)
                        status_color = (0, 255, 255) # Yellow
                    elif distance_mm > 0:
                        status_text = "OPTIMAL DISTANCE ({:.0f}mm)".format(distance_mm)
                        status_color = (0, 255, 0) # Green
                    else:
                        status_text = "Searching for Face..."
                        status_color = (128, 128, 128) # Grey

                    # Draw face mesh (optional, can be removed for cleaner look)
                    # mp_drawing.draw_landmarks(
                    #     frame,
                    #     face_landmarks,
                    #     mp_face_mesh.FACEMESH_TESSELATION,
                    #     drawing_spec,
                    #     drawing_spec
                    # )
                    
                    # Draw a bounding box around the face for clarity
                    # We can use the center landmarks to draw a box relative to the detected face width
                    center_x = int((face_landmarks.landmark[168].x + face_landmarks.landmark[401].x) / 2 * w)
                    center_y = int((face_landmarks.landmark[10].y + face_landmarks.landmark[152].y) / 2 * h)
                    
                    box_size = int(face_width_px * 0.7) # A bit smaller than the face
                    
                    cv2.rectangle(
                        frame,
                        (center_x - box_size, center_y - box_size),
                        (center_x + box_size, center_y + box_size),
                        status_color,
                        2
                    )

            # --- Drawing UI/Stats (The UnboundLocalError Fix) ---
            # These variables now use h and w which are defined on lines 288/289
            
            # Background box for statistics in the lower left corner
            stats_bg_x = 10
            stats_bg_y = h - 130  # CRITICAL: This now uses 'h' which is defined
            stats_bg_w = 400
            stats_bg_h = 120
            
            # Draw the background rectangle
            cv2.rectangle(
                frame,
                (stats_bg_x, stats_bg_y),
                (stats_bg_x + stats_bg_w, stats_bg_y + stats_bg_h),
                (30, 30, 30), # Dark Grey background
                -1
            )
            
            # Title
            cv2.putText(frame, "Eye Distance Monitor", (stats_bg_x + 10, stats_bg_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Status line
            cv2.putText(frame, "STATUS: " + status_text.split('(')[0].strip(), 
                        (stats_bg_x + 10, stats_bg_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        
            # Distance line
            display_dist = "N/A"
            if distance_mm > 0:
                display_dist = f"{distance_mm:.0f} mm"
            cv2.putText(frame, "Current Distance: " + display_dist, 
                        (stats_bg_x + 10, stats_bg_y + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # --- Logging Logic ---
            current_time = time.time()
            if current_time - last_log_time >= LOG_INTERVAL_SEC and distance_mm > 0:
                log_data(distance_mm, status_text.split('(')[0].strip(), face_width_px)
                last_log_time = current_time

            # Display the resulting frame
            cv2.imshow('Eye Monitoring System', frame)
            
            # Exit condition
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped and resources released.")

if __name__ == '__main__':
    # Ensure calibration file exists with a default value if not present
    if not os.path.exists(CALIBRATION_FILE):
        try:
             with open(CALIBRATION_FILE, 'w') as f:
                f.write(str(DEFAULT_FOCAL_LENGTH))
             print(f"Created default {CALIBRATION_FILE} with focal length {DEFAULT_FOCAL_LENGTH}")
        except Exception as e:
            print(f"Error creating default calibration file: {e}")

    start_monitoring()