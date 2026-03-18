from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6 import QtCore

from ..metrics import analyze_metrics
from .config import AppConfig, Thresholds, effective_thresholds, save_config
from .evaluator import PostureEvaluator
from .resources import resource_path


@dataclass(frozen=True)
class EngineFrame:
    bgr: np.ndarray


class PostureEngine(QtCore.QObject):
    frame_ready = QtCore.Signal(object)  # EngineFrame
    metrics_ready = QtCore.Signal(float, float, float)  # gap, tilt, z_depth
    status_ready = QtCore.Signal(str)
    alert_changed = QtCore.Signal(bool)
    calibrated = QtCore.Signal(object)  # Thresholds
    running_changed = QtCore.Signal(bool)
    error = QtCore.Signal(str)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = False
        self._camera_index = cfg.camera_index
        self._show_preview = cfg.show_preview
        self._thresholds = effective_thresholds(cfg)
        self._evaluator = PostureEvaluator()
        self._evaluator.bad_posture_grace_seconds = float(cfg.grace_period_seconds)
        self._evaluator.set_calibrated_thresholds(effective_thresholds(cfg), cfg.is_calibrated)

    @QtCore.Slot()
    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self.running_changed.emit(True)
        threading.Thread(target=self._run_loop, name="PostureEngineThread", daemon=True).start()

    @QtCore.Slot()
    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()

    @QtCore.Slot(int)
    def set_camera_index(self, index: int) -> None:
        with self._lock:
            self._camera_index = int(index)
            self._cfg.camera_index = int(index)
            save_config(self._cfg)

    @QtCore.Slot(bool)
    def set_show_preview(self, enabled: bool) -> None:
        with self._lock:
            self._show_preview = bool(enabled)
            self._cfg.show_preview = bool(enabled)
            save_config(self._cfg)

    @QtCore.Slot(float)
    def set_grace_period_seconds(self, seconds: float) -> None:
        value = max(0.0, float(seconds))
        with self._lock:
            self._cfg.grace_period_seconds = value
            self._evaluator.bad_posture_grace_seconds = value
            save_config(self._cfg)

    @QtCore.Slot(bool)
    def set_use_manual_thresholds(self, enabled: bool) -> None:
        with self._lock:
            self._cfg.use_manual_thresholds = bool(enabled)
            self._thresholds = effective_thresholds(self._cfg)
            save_config(self._cfg)

    @QtCore.Slot(float, float, float)
    def set_manual_thresholds(self, gap: float, z: float, tilt: float) -> None:
        with self._lock:
            self._cfg.manual_thresholds = Thresholds(gap=float(gap), z=float(z), tilt=float(tilt))
            if self._cfg.use_manual_thresholds:
                self._thresholds = effective_thresholds(self._cfg)
            save_config(self._cfg)

    @QtCore.Slot(float, float, float)
    def set_calibrated_thresholds(self, gap: float, z: float, tilt: float) -> None:
        with self._lock:
            calibrated = Thresholds(gap=float(gap), z=float(z), tilt=float(tilt))
            self._cfg.calibrated_thresholds = calibrated
            # Keep manual defaults in sync with calibration so users can tweak from baseline.
            self._cfg.manual_thresholds = calibrated
            self._cfg.is_calibrated = True
            # Refresh active thresholds regardless of mode since manual defaults may change.
            self._thresholds = effective_thresholds(self._cfg)
            save_config(self._cfg)

    @QtCore.Slot()
    def start_calibration(self) -> None:
        self._evaluator.start_calibration()

    def _run_loop(self) -> None:
        BaseOptions = python.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        last_alert: Optional[bool] = None

        try:
            model_path = resource_path("pose_landmarker_full.task")
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=self._make_callback(),
            )

            cap = cv.VideoCapture(self._camera_index)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera.")

            with PoseLandmarker.create_from_options(options) as landmarker:
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.02)
                        continue

                    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    landmarker.detect_async(mp_image, int(time.time() * 1000))

                    with self._lock:
                        show_preview = self._show_preview

                    if show_preview:
                        self.frame_ready.emit(EngineFrame(bgr=frame))

                    time.sleep(0.001)

            cap.release()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._running = False
            self.running_changed.emit(False)

    def _make_callback(self):
        PoseLandmarkerResult = vision.PoseLandmarkerResult

        def _cb(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
            if not result.pose_landmarks:
                return
            pose_landmarks = result.pose_landmarks[0]
            gap, tilt, z_depth = analyze_metrics(pose_landmarks)
            self.metrics_ready.emit(gap, tilt, z_depth)

            with self._lock:
                thresholds = self._thresholds

            out = self._evaluator.update(gap=gap, tilt=tilt, z_depth=z_depth, thresholds=thresholds)
            self.status_ready.emit(out.posture_status)
            self.alert_changed.emit(out.alert_active)

            if out.calibrated_thresholds is not None:
                calibrated = out.calibrated_thresholds
                self.calibrated.emit(calibrated)
                self.set_calibrated_thresholds(calibrated.gap, calibrated.z, calibrated.tilt)

        return _cb

