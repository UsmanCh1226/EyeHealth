import cv2
import dlib
from scipy.spatial import distance as dist
import time
import numpy as np

# --- CONSTANTS ---

# Eye Aspect Ratio (EAR) settings for blink detection
EAR_THRESHOLD = 0.28
CONSECUTIVE_FRAMES = 3

# 20-20-20 Rule settings (Set to shorter values for easy testing)
WORK_INTERVAL_SEC = 20  # Change to 20 * 60 for 20 minutes
REST_DURATION_SEC = 5   # Change to 20 for the full 20-second rest

# Font and color settings for text display
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_GREEN = (0, 255, 0)
TEXT_COLOR_RED = (0, 0, 255)
ALERT_BG_COLOR = (20, 20, 20) # Dark gray background for alerts

# Dlib's facial landmark indices for the left and right eye
# Used for calculating the Eye Aspect Ratio (EAR)
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

# --- FUNCTIONS ---

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) to determine if an eye is open or closed.
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmarks (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def draw_text_with_background(frame, text, position, font_scale, color, thickness, bg_color, padding=5):
    """Draws text on the frame with a solid background for better visibility."""
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    
    # Calculate background rectangle coordinates
    x, y = position
    pt1 = (x, y - text_height - baseline - padding)
    pt2 = (x + text_width + padding * 2, y + padding)
    
    # Draw the background rectangle
    cv2.rectangle(frame, pt1, pt2, bg_color, -1)
    
    # Draw the text
    cv2.putText(frame, text, (x + padding, y - baseline), FONT, font_scale, color, thickness, cv2.LINE_AA)


# --- MAIN APPLICATION LOGIC ---

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # NOTE: You must have this file downloaded

# Initialize the video stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Blink Counters
TOTAL_BLINKS = 0
FRAME_COUNTER = 0

# Time trackers for 20-20-20 rule
start_time = time.time()
is_resting = False
rest_start_time = 0
elapsed_work_time = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Check 20-20-20 Timer
    if not is_resting:
        elapsed_work_time = time.time() - start_time
        if elapsed_work_time >= WORK_INTERVAL_SEC:
            is_resting = True
            rest_start_time = time.time()
            # Stop blink counting during rest phase
        
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
        
        # Center the alert message on the screen
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h//2 - 100), (w, h//2 + 100), TEXT_COLOR_RED, -1)
        
        draw_text_with_background(frame, alert_text, (20, h//2 - 20), 1.0, TEXT_COLOR_WHITE, 2, TEXT_COLOR_RED, padding=10)
        draw_text_with_background(frame, rest_timer_text, (20, h//2 + 50), 1.2, TEXT_COLOR_WHITE, 3, TEXT_COLOR_RED, padding=10)


        if elapsed_rest_time >= REST_DURATION_SEC:
            # Rest period is over, reset the timer
            is_resting = False
            start_time = time.time()
            elapsed_work_time = 0 # Reset work time tracker

    # Only process blinks and draw landmarks if the user is not actively resting
    if not is_resting and len(rects) > 0:
        for rect in rects:
            # Determine the facial landmarks for the face region
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # Extract the left and right eye coordinates
            leftEye = shape[L_START:L_END]
            rightEye = shape[R_START:R_END]

            # Compute the EAR for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            
            # Use the convex hull to visualize the eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, TEXT_COLOR_GREEN, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, TEXT_COLOR_GREEN, 1)

            # Check to see if the eye aspect ratio is below the blink threshold
            if ear < EAR_THRESHOLD:
                FRAME_COUNTER += 1
            else:
                # If the eyes were closed for a sufficient number of frames, increment the total number of blinks
                if FRAME_COUNTER >= CONSECUTIVE_FRAMES:
                    TOTAL_BLINKS += 1
                
                # Reset the frame counter
                FRAME_COUNTER = 0

            # Display the EAR value and Blink Counter on the frame
            draw_text_with_background(frame, f"EAR: {ear:.2f}", (10, 30), 0.7, TEXT_COLOR_WHITE, 2, ALERT_BG_COLOR, padding=3)
            draw_text_with_background(frame, f"Blinks: {TOTAL_BLINKS}", (10, 60), 0.7, TEXT_COLOR_WHITE, 2, ALERT_BG_COLOR, padding=3)
            
    # Show the frame
    cv2.imshow("Eye Monitor", frame)

    # If the 'q' key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()