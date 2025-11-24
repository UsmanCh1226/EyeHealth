import numpy as np
from scipy.spatial import distance as dist
import cv2 # <-- CORRECTED: cv2 import is now at the top
import dlib # <-- Added for consistency, though only used conceptually

# ==============================================================================
# 1. CONSTANTS AND LANDMARK INDICES
# ==============================================================================

# Dlib's 68-point facial landmark indices for the left and right eyes
# These indices are critical for calculating the Eye Aspect Ratio (EAR).
LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))

# ==============================================================================
# 2. CORE GEOMETRY & CV METRICS
# ==============================================================================

def calculate_ear(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR).

    The EAR is a measure of the eye openness, calculated as the ratio of
    the vertical distances between eye landmarks to the horizontal distance.
    

    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye_points (np.array): A 6x2 array of (x, y) coordinates for the eye landmarks.
                               Points must be ordered (p1 to p6).

    Returns:
        float: The Eye Aspect Ratio. Returns 0.0 if input is invalid.
    """
    if len(eye_points) != 6:
        return 0.0

    # A: Vertical distance between the two vertical landmarks (p2 and p6)
    A = dist.euclidean(eye_points[1], eye_points[5])
    # B: Vertical distance between the other two vertical landmarks (p3 and p5)
    B = dist.euclidean(eye_points[2], eye_points[4])

    # C: Horizontal distance (p1 and p4)
    C = dist.euclidean(eye_points[0], eye_points[3])

    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

def euclidean_distance(ptA, ptB):
    """
    Simple wrapper for calculating the Euclidean distance between two points.
    """
    return dist.euclidean(ptA, ptB)

def midpoint(ptA, ptB):
    """
    Calculates the midpoint between two (x, y) points.
    """
    return ((ptA[0] + ptB[0]) // 2, (ptA[1] + ptB[1]) // 2)

# ==============================================================================
# 3. HELPER FUNCTIONS FOR RENDERING
# ==============================================================================

def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    """
    Conceptual function to draw the detected landmarks on the frame for visualization
    using cv2 (OpenCV).
    """
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, color, -1)

def draw_info(frame, text, location=(10, 30), color=(0, 0, 255)):
    """
    Conceptual function to display diagnostic text on the frame using cv2.
    """
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


if __name__ == '__main__':
    print("--- CV Utilities Loaded ---")
    print(f"Left Eye Indices: {LEFT_EYE_INDICES}")
    print(f"Right Eye Indices: {RIGHT_EYE_INDICES}")

    # Example usage (simulated)
    # Define an open eye (large vertical, large horizontal)
    open_eye = np.array([
        [10, 10], [11, 20], [12, 20],
        [30, 10],
        [12, 0], [11, 0]
    ])
    ear_open = calculate_ear(open_eye)
    print(f"Simulated Open EAR: {ear_open:.4f} (should be higher)")