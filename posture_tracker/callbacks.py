from __future__ import annotations

import time
from typing import Callable

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from .metrics import analyze_metrics
from .state import PostureState


PoseLandmarkerResult = vision.PoseLandmarkerResult


def make_result_callback(state: PostureState) -> Callable[[PoseLandmarkerResult, mp.Image, int], None]:
    def result_callback(
        result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        frame_rgb = output_image.numpy_view()
        annotated_image = np.copy(frame_rgb)

        if result.pose_landmarks:
            pose_landmarks = result.pose_landmarks[0]
            gap, tilt, z_depth = analyze_metrics(pose_landmarks)
            state.current_metrics.update({"gap": gap, "tilt": tilt, "z_depth": z_depth})

            # --- STEP 2: AUTO-CALIBRATION LOGIC ---
            if not state.is_calibrated:
                state.posture_status = (
                    f"CALIBRATING... Hold Still ({len(state.calibration_data)}/30)"
                )
                state.calibration_data.append((gap, z_depth))
                if len(state.calibration_data) >= 30:
                    avg_cal_gap = sum(g for g, z in state.calibration_data) / 30
                    avg_cal_z = sum(z for g, z in state.calibration_data) / 30
                    # Set thresholds slightly more aggressive than baseline
                    state.thresholds["gap"] = avg_cal_gap * 0.85
                    state.thresholds["z"] = avg_cal_z * 1.30
                    state.is_calibrated = True

            # --- STEP 1: DETECTION & TIMER LOGIC ---
            else:
                is_bad = (
                    (z_depth < state.thresholds["z"])
                    or (gap < state.thresholds["gap"])
                    or (tilt > 0.06)
                )

                if is_bad:
                    if state.bad_posture_start_time is None:
                        state.bad_posture_start_time = time.time()

                    elapsed = time.time() - state.bad_posture_start_time
                    if elapsed > 3.0:
                        state.posture_status = f"WARNING: FIX POSTURE! ({int(elapsed)}s)"
                        state.alert_active = True
                    else:
                        state.posture_status = "Good (grace period)"
                        state.alert_active = False
                else:
                    state.posture_status = "Good Posture"
                    state.bad_posture_start_time = None
                    state.alert_active = False

            # Draw skeleton
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            )

        state.latest_annotated_frame = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)

    return result_callback

