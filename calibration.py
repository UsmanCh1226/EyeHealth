import cv2
import dlib
import numpy as np
import json
import time
from scipy.spatial import distance as dist
from constants import KNOWN_FACE_WIDTH_CM, KNOWN_DISTANCE_CM, R_START, R_END, L_START, L_END, CALIBRATION_FILE_PATH, FONT, TEXT_COLOR_WHITE, TEXT_COLOR_GREEN, TEXT_COLOR_RED
from utils import focal_length_finder, draw_text_with_background

# --- INITIALIZATION ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream for calibration.")
    exit()

def get_face_width(shape):
    """Calculates the width of the face (distance between temples)."""
    # Assuming face landmarks 1 (left temple) and 15 (right temple) or a similar pair
    left_cheek = shape[2] 
    right_cheek = shape[14] 
    return dist.euclidean(left_cheek, right_cheek)

def calculate_focal_length():
    """Main function to run the calibration process."""
    print("--- Starting Camera Calibration ---")
    print(f"Goal: Measure the camera's focal length using a known distance of {KNOWN_DISTANCE_CM} cm.")
    print("Please sit exactly 60 cm away from your screen and look straight at the camera.")
    print("Press 'S' to save the calibration data.")
    
    pixel_widths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        status_text = "Adjust distance to 60 cm..."
        color = TEXT_COLOR_RED
        
        if len(rects) > 0:
            rect = rects[0]
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            face_pixel_width = get_face_width(shape)
            
            # Draw face rectangle
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), TEXT_COLOR_GREEN, 2)
            
            status_text = f"Current Face Pixel Width: {face_pixel_width:.2f}"
            color = TEXT_COLOR_GREEN
            
            # Collect data points
            if face_pixel_width > 50: # Simple sanity check
                pixel_widths.append(face_pixel_width)

        # Display instructions
        draw_text_with_background(frame, status_text, (20, 50), 0.7, color, 2, (20, 20, 20))
        draw_text_with_background(frame, f"Data Points Collected: {len(pixel_widths)}", (20, 100), 0.7, TEXT_COLOR_WHITE, 2, (20, 20, 20))
        draw_text_with_background(frame, "Press 'S' to SAVE / 'Q' to QUIT", (20, frame.shape[0] - 30), 0.7, TEXT_COLOR_WHITE, 2, (20, 20, 20))

        cv2.imshow("Calibration Tool", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(pixel_widths) > 10: # Require at least 10 samples
            break
        elif key == ord('s') and len(pixel_widths) <= 10:
             print("Need more data points (at least 10) before saving.")


    # --- FINAL CALCULATION AND SAVE ---
    if len(pixel_widths) > 0:
        avg_pixel_width = np.median(pixel_widths) # Use median for robustness
        FOCAL_LENGTH = focal_length_finder(KNOWN_DISTANCE_CM, KNOWN_FACE_WIDTH_CM, avg_pixel_width)
        
        # Save focal length
        calibration_data = {
            "focal_length": FOCAL_LENGTH,
            "known_distance_cm": KNOWN_DISTANCE_CM,
            "known_face_width_cm": KNOWN_FACE_WIDTH_CM,
            "calibration_timestamp": time.time()
        }
        
        with open(CALIBRATION_FILE_PATH, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print("\n--- CALIBRATION COMPLETE ---")
        print(f"Average Pixel Width: {avg_pixel_width:.2f}")
        print(f"Calculated Focal Length: {FOCAL_LENGTH:.2f}")
        print(f"Calibration saved to {CALIBRATION_FILE_PATH}. This will be used by the main monitor.")
    else:
        print("Calibration failed. No face detected or not enough data points.")


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    calculate_focal_length()