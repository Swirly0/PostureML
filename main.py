import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import time

# Setup Aliases
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
VisionRunningMode = vision.RunningMode

# Global States
latest_annotated_frame = None
current_landmarks = None
posture_status = "Scanning..."
current_metrics = {"gap": 0, "tilt": 0, "z_depth": 0}

def analyze_metrics(landmarks):
    if not landmarks:
        return 0, 0, 0
    
    # 1. Ear-to-Shoulder Gap (Vertical)
    left_gap = landmarks[11].y - landmarks[7].y
    right_gap = landmarks[12].y - landmarks[8].y
    avg_gap = (left_gap + right_gap) / 2

    # 2. Shoulder Tilt
    shoulder_tilt = abs(landmarks[11].y - landmarks[12].y)

    # 3. Nose Z-Depth (Forward Lean)
    nose_z = landmarks[0].z
    
    return avg_gap, shoulder_tilt, nose_z

def result_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame, current_landmarks, posture_status, current_metrics
    
    frame_rgb = output_image.numpy_view()
    annotated_image = np.copy(frame_rgb)

    if result.pose_landmarks:
        pose_landmarks = result.pose_landmarks[0]
        current_landmarks = pose_landmarks 
        
        gap, tilt, z_depth = analyze_metrics(pose_landmarks)
        current_metrics["gap"] = gap
        current_metrics["tilt"] = tilt
        current_metrics["z_depth"] = z_depth

        # --- SENSITIVITY THRESHOLDS ---
        # 1. Forward Lean (Z-Depth): If nose gets closer than -1.1 (Your slouch was -1.39)
        if z_depth < -1.10:
            posture_status = "Warning: Leaning Forward!"
        # 2. Slouching (Gap): If gap drops below 0.24 (Your good was 0.27, bad was 0.22)
        elif gap < 0.24:
            posture_status = "Warning: Sit Up Straight!"
        # 3. Tilt: If shoulders uneven
        elif tilt > 0.04:
            posture_status = "Warning: Uneven Shoulders!"
        else:
            posture_status = "Good Posture"

        # Draw skeleton
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
    
    latest_annotated_frame = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)

def main():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback)

    cap = cv.VideoCapture(0)

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
            
            display_frame = latest_annotated_frame if latest_annotated_frame is not None else frame
            
            # Visual Feedback
            color = (0, 255, 0) if "Good" in posture_status else (0, 0, 255)
            cv.putText(display_frame, f"STATUS: {posture_status}", (20, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Subtle HUD for live metrics
            cv.putText(display_frame, f"Z-Depth: {current_metrics['z_depth']:.2f}", (20, 80), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv.imshow('Posture Correction', display_frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                print(f"DEBUG | Gap: {current_metrics['gap']:.4f} | Z: {current_metrics['z_depth']:.4f}")
            elif key == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()