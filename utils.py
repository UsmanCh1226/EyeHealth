import cv2
from scipy.spatial import distance as dist
from constants import FONT, ALERT_BG_COLOR

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR) for blink detection."""
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    # EAR calculation
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
    """Calculates the Focal Length of the camera based on calibration."""
    focal_length = (pixel_width * known_distance) / known_width
    return focal_length

def distance_finder(focal_length, known_width, pixel_width):
    """Calculates the distance (cm) from the camera to the object."""
    distance = (known_width * focal_length) / pixel_width
    return distance