from __future__ import annotations

import time

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .callbacks import make_result_callback
from .state import PostureState


# Setup Aliases (kept to mirror original script structure)
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode


def run_posture_tracker() -> None:
    state = PostureState()
    result_callback = make_result_callback(state)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
    )

    cap = cv.VideoCapture(0)

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            display_frame = (
                state.latest_annotated_frame
                if state.latest_annotated_frame is not None
                else frame
            )

            # UI Styling
            text_color = (0, 0, 255) if state.alert_active else (0, 255, 0)
            if "CALIBRATING" in state.posture_status:
                text_color = (0, 255, 255)  # Yellow

            cv.putText(
                display_frame,
                state.posture_status,
                (20, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
            )

            cv.imshow("Smart Posture Tracker", display_frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()

