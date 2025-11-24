import cv2
import dlib
from scipy.spatial import distance as dist
import time
import numpy as np

# --- CONSTANTS ---

# Eye Aspect Ratio (EAR) settings for blink detection
EAR_THRESHOLD = 0.28
CONSECUTIVE_FRAMES = 3 # Frames for a normal blink to be counted

# DROWSINESS/FATIGUE DETECTION setting
# If eyes are closed for this many consecutive frames, trigger a fatigue alert.
DROWSINESS_FRAMES = 45 # Approximately 1.5 seconds at 30 FPS

# 20-20-20 Rule settings
WORK_INTERVAL_SEC = 20  # Change to 20 * 60 for 20 minutes
REST_DURATION_SEC = 5   # Change to 20 for the full 20-second rest

# Distance Monitoring Calibration (IMPORTANT: These values might need adjustment for your specific camera/setup)
KNOWN_FACE_WIDTH_CM = 14.0 
KNOWN_DISTANCE_CM = 60.0
PIXEL_WIDTH_AT_KNOWN_DISTANCE = 300.0 

# Recommended minimum distance for eye health
MIN_HEALTHY_DISTANCE_CM = 50.0 # Alert if closer than 50 cm

# Font and color settings for text display
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_GREEN = (0, 255, 0)
TEXT_COLOR_RED = (0, 0, 255)
ALERT_BG_COLOR = (20, 20, 20) # Dark gray background for non-critical text
FATIGUE_BG_COLOR = (0, 0, 255) # Blue background for fatigue warning

# Dlib's facial landmark indices for the left and right eye
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

# --- UTILITY FUNCTIONS ---

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) to determine if an eye is open or closed.
    """
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def draw_text_with_background(frame, text, position, font_scale, color, thickness, bg_color, padding=5):
    """Draws text on the frame with a solid background for better visibility."""
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = position
    pt1 = (x, y - text_height - baseline - padding)
    pt2 = (x + text_width + padding * 2, y + padding)
    cv2.rectangle(frame, pt1, pt2, bg_color, -1)
    cv2.putText(frame, text, (x + padding, y - baseline), FONT, font_scale, color, thickness, cv2.LINE_AA)
    
def focal_length_finder(known_distance, known_width, pixel_width):
    """Calculates the Focal Length of the camera."""
    focal_length = (pixel_width * known_distance) / known_width
    return focal_length

def distance_finder(focal_length, known_width, pixel_width):
    """Calculates the distance (cm) from the camera to the object."""
    distance = (known_width * focal_length) / pixel_width
    return distance

# --- MAIN APPLICATION LOGIC ---

# Initialize Dlib's components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # NOTE: You must have this file downloaded

# Initialize the video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Blink Counters
TOTAL_BLINKS = 0
FRAME_COUNTER = 0 # Used for both blink and drowsiness tracking
is_drowsy_alert = False

# Time trackers for 20-20-20 rule
start_time = time.time()
is_resting = False
rest_start_time = 0
elapsed_work_time = 0

# Focal Length Calculation
FOCAL_LENGTH = focal_length_finder(KNOWN_DISTANCE_CM, KNOWN_FACE_WIDTH_CM, PIXEL_WIDTH_AT_KNOWN_DISTANCE)
print(f"Calculated Focal Length: {FOCAL_LENGTH:.2f} pixels.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    current_distance_cm = -1
    distance_alert_active = False
    
    # Reset drowsiness alert if it was active
    is_drowsy_alert = False

    # --- 1. POSTURE/DISTANCE MONITORING ---
    if len(rects) > 0:
        rect = rects[0] 
        
        # Get the pixel width of the detected face (width of the bounding box)
        face_pixel_width = rect.right() - rect.left()
        
        # Calculate the real-time distance
        current_distance_cm = distance_finder(FOCAL_LENGTH, KNOWN_FACE_WIDTH_CM, face_pixel_width)

        # Check if the user is too close
        if current_distance_cm < MIN_HEALTHY_DISTANCE_CM:
            distance_alert_active = True
            
            # Draw a bounding box around the face in RED if too close
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), TEXT_COLOR_RED, 2)
            
            # Display the alert message
            alert_msg = f"TOO CLOSE! Maintain > {MIN_HEALTHY_DISTANCE_CM:.0f} cm"
            draw_text_with_background(frame, alert_msg, (10, 90), 0.7, TEXT_COLOR_WHITE, 2, TEXT_COLOR_RED, padding=3)

        else:
            # Draw a bounding box around the face in GREEN if distance is good
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), TEXT_COLOR_GREEN, 2)
        
        # Display the current distance
        distance_text = f"Distance: {current_distance_cm:.0f} cm"
        color = TEXT_COLOR_RED if distance_alert_active else TEXT_COLOR_WHITE
        draw_text_with_background(frame, distance_text, (frame.shape[1] - 220, 60), 0.7, color, 2, ALERT_BG_COLOR, padding=3)
        
        # --- 2. BLINK/EAR LOGIC & DROWSINESS CHECK ---
        
        # Only process landmarks for the first face 
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape[L_START:L_END]
        rightEye = shape[R_START:R_END]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Use the convex hull to visualize the eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, TEXT_COLOR_GREEN, 1)
        cv2.drawContours(frame, [rightEyeHull], -1, TEXT_COLOR_GREEN, 1)

        # Blink and Drowsiness Counting Logic
        if ear < EAR_THRESHOLD:
            FRAME_COUNTER += 1
            
            # Check for DROWSINESS (Eyes closed too long)
            if FRAME_COUNTER >= DROWSINESS_FRAMES and not is_resting:
                is_drowsy_alert = True

        else:
            # Eyes are open
            if FRAME_COUNTER >= CONSECUTIVE_FRAMES and FRAME_COUNTER < DROWSINESS_FRAMES:
                # Count as a regular blink
                TOTAL_BLINKS += 1
            
            # Reset the frame counter
            FRAME_COUNTER = 0

        # Display the EAR value and Blink Counter on the frame
        draw_text_with_background(frame, f"EAR: {ear:.2f}", (10, 30), 0.7, TEXT_COLOR_WHITE, 2, ALERT_BG_COLOR, padding=3)
        draw_text_with_background(frame, f"Blinks: {TOTAL_BLINKS}", (10, 60), 0.7, TEXT_COLOR_WHITE, 2, ALERT_BG_COLOR, padding=3)

    # --- 3. 20-20-20 TIMER LOGIC ---
    if not is_resting:
        elapsed_work_time = time.time() - start_time
        if elapsed_work_time >= WORK_INTERVAL_SEC:
            is_resting = True
            rest_start_time = time.time()
        
        # Display elapsed time
        minutes = int(elapsed_work_time // 60)
        seconds = int(elapsed_work_time % 60)
        timer_text = f"Work Time: {minutes:02d}:{seconds:02d}"
        draw_text_with_background(frame, timer_text, (frame.shape[1] - 220, 30), 0.7, TEXT_COLOR_WHITE, 2, ALERT_BG_COLOR, padding=3)

    else:
        # User is in the rest phase
        elapsed_rest_time = time.time() - rest_start_time
        remaining_rest = max(0, REST_DURATION_SEC - elapsed_rest_time)

        # Display REST ALERT
        alert_text = "20-20-20 Break! Look at least 20 feet away."
        rest_timer_text = f"RESTING: {int(remaining_rest)}s Remaining"
        
        h, w = frame.shape[:2]
        # Draw a large red rectangle for the rest break
        cv2.rectangle(frame, (0, h//2 - 100), (w, h//2 + 100), TEXT_COLOR_RED, -1)
        
        draw_text_with_background(frame, alert_text, (20, h//2 - 20), 1.0, TEXT_COLOR_WHITE, 2, TEXT_COLOR_RED, padding=10)
        draw_text_with_background(frame, rest_timer_text, (20, h//2 + 50), 1.2, TEXT_COLOR_WHITE, 3, TEXT_COLOR_RED, padding=10)

        if elapsed_rest_time >= REST_DURATION_SEC:
            is_resting = False
            start_time = time.time()
            elapsed_work_time = 0
            
    # --- 4. DROWSINESS ALERT DISPLAY ---
    if is_drowsy_alert:
        h, w = frame.shape[:2]
        fatigue_alert_text = "FATIGUE ALERT! Take a break."
        # Flash the alert in a prominent position
        draw_text_with_background(frame, fatigue_alert_text, (w//2 - 250, h - 50), 1.5, TEXT_COLOR_WHITE, 3, FATIGUE_BG_COLOR, padding=15)
            
    # Show the frame
    cv2.imshow("Eye Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()