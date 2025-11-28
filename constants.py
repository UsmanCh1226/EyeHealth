import cv2

# --- CALIBRATION AND LOGGING ---
CALIBRATION_FILE_PATH = "calibration.json"
LOG_FILE_PATH = "eye_health_log.csv"
LOGGING_INTERVAL_SEC = 15 # Log data every 15 seconds

# --- CAMERA PARAMETERS ---
# Known physical width of the face, used for distance calculation. 
KNOWN_FACE_WIDTH_CM = 6.5 
FOCAL_LENGTH = 0.0 # Placeholder, will be loaded from calibration.json

# --- THRESHOLDS ---
EAR_THRESHOLD = 0.25      # Eye Aspect Ratio threshold for drowsiness/blinks
MAR_THRESHOLD = 0.70      # Mouth Aspect Ratio threshold for yawns
DIST_THRESHOLD_CM = 45    # Distance threshold for proximity alert (45 cm or less is too close)

# --- VISUALIZATION & COLORS ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR_WHITE = (255, 255, 255) 
TEXT_COLOR_GREEN = (0, 255, 0)
TEXT_COLOR_RED = (0, 0, 255)
TEXT_COLOR_YELLOW = (0, 255, 255)

# Background colors for alerts
ALERT_BG_COLOR = (20, 20, 160) # Dark Red/Blue for high alert
NORMAL_BG_COLOR = (50, 50, 50) # Dark Gray for normal display

# --- FACIAL LANDMARK INDICES (MediaPipe Face Mesh) ---
L_START, L_END = 362, 263
R_START, R_END = 133, 33

# Mouth landmarks for MAR calculation
MOUTH_TOP = 13     # Top lip (Center)
<<<<<<< HEAD
MOUTH_BOTTOM = 14  # Bottom lip (Center)
=======
MOUTH_BOTTOM = 14  # Bottom lip (Center)
>>>>>>> 8b51a21e0bba80a34cfbe030a8ec59ff159d9213
