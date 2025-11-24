import cv2
import dlib
import numpy as np
import time
from src.cv.utils_cv import calculate_ear, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, draw_info

# ==============================================================================
# 1. CONSTANTS AND THRESHOLDS
# ==============================================================================

# Threshold for Eye Aspect Ratio (EAR) to classify a blink.
# If EAR falls below this value, the eye is considered closed.
EYE_AR_THRESH = 0.25 

# Consecutive frames the eye must be below the threshold to officially register a blink.
# This prevents false positives due to noise in the image.
EYE_AR_CONSEC_FRAMES = 3 

# Time in seconds for a user to be considered drowsy (e.g., 3 seconds of continuous closure)
DROWSINESS_TIME_THRESH = 3.0 

# ==============================================================================
# 2. STATE MANAGEMENT CLASS
# ==============================================================================

class DrowsinessDetector:
    """
    Manages the state required for blink counting and drowsiness detection.
    """
    def __init__(self):
        # Frame counter for how long the eyes have been below the EAR threshold
        self.frame_counter = 0
        # Total number of blinks detected
        self.total_blinks = 0
        # Time when the continuous eye closure started (for drowsiness)
        self.start_closure_time = None
        # Flag indicating if a drowsiness alert should be active
        self.is_drowsy_alert = False

    def process_frame(self, frame, landmarks):
        """
        Processes a single frame to detect blinks and potential drowsiness.

        Args:
            frame (np.array): The current video frame.
            landmarks (list): List of (x, y) coordinates for all 68 facial landmarks.

        Returns:
            np.array: The frame with overlayed information.
            dict: Current detection state (EAR, blinks, alert status).
        """
        # 1. Extract eye coordinates using utility indices
        left_eye = np.array([landmarks[i] for i in LEFT_EYE_INDICES])
        right_eye = np.array([landmarks[i] for i in RIGHT_EYE_INDICES])

        # 2. Calculate average EAR
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # 3. Check for Eye Closure (Blink/Drowsiness)
        if avg_ear < EYE_AR_THRESH:
            # Eyes are closed: increment frame counter
            self.frame_counter += 1
            
            # Start tracking closure time if not already started
            if self.start_closure_time is None:
                self.start_closure_time = time.time()

            # Check for Drowsiness Alert (sustained closure)
            if self.start_closure_time and (time.time() - self.start_closure_time) > DROWSINESS_TIME_THRESH:
                self.is_drowsy_alert = True
                draw_info(frame, "!!! DROWSINESS ALERT !!!", (200, 60), (0, 0, 255))
            else:
                draw_info(frame, "Eyes Closed", (200, 30), (0, 255, 255))

        else:
            # Eyes are open: reset timers and check if a blink just finished
            
            # If the eye was closed for a sufficient number of frames, register a blink
            if self.frame_counter >= EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
            
            # Reset counters/timers
            self.frame_counter = 0
            self.start_closure_time = None
            self.is_drowsy_alert = False

        # 4. Draw information on the frame
        draw_info(frame, f"EAR: {avg_ear:.2f}", (10, 30), (255, 255, 0))
        draw_info(frame, f"Blinks: {self.total_blinks}", (10, 60), (255, 0, 0))

        # 5. Return processed frame and state
        state = {
            'avg_ear': avg_ear,
            'total_blinks': self.total_blinks,
            'is_drowsy_alert': self.is_drowsy_alert
        }
        return frame, state

# ==============================================================================
# 3. CONCEPTUAL MAIN LOOP (Requires Dlib and Webcam)
# ==============================================================================

def run_drowsiness_test():
    """
    Conceptual function to test the DrowsinessDetector class.
    NOTE: Requires Dlib's 'shape_predictor_68_face_landmarks.dat' and a webcam.
    """
    # Initialize the required dlib components here if running independently
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")

    detector = None # Placeholder
    predictor = None # Placeholder

    if detector is None:
        print("--- Drowsiness Test: Conceptual Mode ---")
        print("Requires Dlib and webcam to run live. Logic implemented.")
        return

    # detector_instance = DrowsinessDetector()
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret: break

    #     # Placeholder for detection and landmark finding
    #     # rects = detector(frame, 0)
    #     # if len(rects) > 0:
    #     #     shape = predictor(frame, rects[0])
    #     #     landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    #     #     processed_frame, state = detector_instance.process_frame(frame, landmarks)
    #     # else:
    #     #     processed_frame = frame

    #     # cv2.imshow('Drowsiness Monitor', processed_frame)
    #     # if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    # cap.release()
    # cv2.destroyAllWindows()
    print("Drowsiness detection logic ready.")


if __name__ == '__main__':
    run_drowsiness_test()