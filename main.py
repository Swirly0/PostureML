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

latest_annotated_frame = None
face_landmarks = None

def calculatePosture(landmarks):
    #distnace from ears to the shoulders
    left_distance = abs(landmarks[8].y - landmarks[12].y)
    right_distance = abs(landmarks[7].y - landmarks[11].y)
    avg_gap = (left_distance + right_distance) / 2
    # nose_distance
    faceZ = landmarks[0].z

    shoulder_slope = abs((landmarks[12].y - landmarks[11].y) / (abs(landmarks[12].x) - landmarks[11].x))
    print(f"LEFT EAR HEGHT = {landmarks[8].y}, RIGHT EAR: {landmarks[7].y}")
    print(f"LEFT SHOULDER = {landmarks[7].y}, RIGHT SHOULDER: {landmarks[11].y}")
    print(left_distance, right_distance, shoulder_slope, avg_gap)

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame
    global face_landmarks
    # Get the frame from MediaPipe (RGB)
    frame_rgb = output_image.numpy_view()
    # Create a copy to draw on
    annotated_image = np.copy(frame_rgb)
    if result.pose_landmarks:
        face_landmarks = result.pose_world_landmarks[0][:13]
        # Draw the landmarks using the Tasks drawing_utils
        for pose_landmarks in result.pose_landmarks:
            #face and shoulders
            # face_landmarks = pose_landmarks[:13]
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                # In Tasks, connections are found here:
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
    
    # Convert to BGR for OpenCV display
    latest_annotated_frame = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)

# Configuration


def main():
    options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

    cap = cv.VideoCapture(0)

    with PoseLandmarker.create_from_options(options) as landmarker:
        landmarks = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
            
            if latest_annotated_frame is not None:
                cv.imshow('Posture detection', latest_annotated_frame)
            else:
                cv.imshow('Posture detection', frame)
            if cv.waitKey(1) == ord('c'):
                landmarks = face_landmarks
                calculatePosture(landmarks)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()