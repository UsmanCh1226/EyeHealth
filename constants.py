import cv2

# --- CALIBRATION CONSTANTS ---
# Standard measured distance and corresponding object width (e.g., face width)
KNOWN_FACE_WIDTH_CM = 14.0  # Average width of a human head/face in cm
KNOWN_DISTANCE_CM = 50.0    # Distance in cm used during calibration (e.g., arms length)
CALIBRATION_FILE_PATH = "calibration.json"

# --- LOGGING CONSTANTS ---
LOG_FILE_PATH = "eye_health_log.csv"
LOGGING_INTERVAL_SEC = 15  # Log metrics every 15 seconds

# --- MONITORING THRESHOLDS & SETTINGS ---
EAR_THRESHOLD = 0.23      # Eye Aspect Ratio threshold for blink/drowsiness detection
MAR_THRESHOLD = 0.50      # Mouth Aspect Ratio threshold for yawning detection
FRAME_RATE = 30           # Expected camera frame rate (affects blink calculation)
DIST_THRESHOLD_CM = 45.0  # Maximum recommended working distance (e.g., 45-50 cm is healthy)

# --- VISUAL CONSTANTS (Colors and Fonts) ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_GREEN = (0, 255, 0)
TEXT_COLOR_RED = (0, 0, 255)
TEXT_COLOR_YELLOW = (0, 255, 255)

# Background Colors for drawn text boxes
ALERT_BG_COLOR = (0, 165, 255)  # Orange/Red (BGR) for warnings
NORMAL_BG_COLOR = (50, 50, 50)  # Dark Gray (BGR) for non-alert info

# --- FACE MESH LANDMARK INDICES (Based on MediaPipe) ---
# Left Eye Landmarks (Indices defining the perimeter)
L_START = 362
L_END = 398
# Right Eye Landmarks (Indices defining the perimeter)
R_START = 133
R_END = 159
# Mouth Landmarks for Yawning (simplified indices)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291