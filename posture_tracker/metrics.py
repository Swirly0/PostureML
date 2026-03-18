from __future__ import annotations

from typing import Sequence, Tuple


def analyze_metrics(landmarks: Sequence) -> Tuple[float, float, float]:
    if not landmarks:
        return 0, 0, 0
    left_gap = landmarks[11].y - landmarks[7].y
    right_gap = landmarks[12].y - landmarks[8].y
    avg_gap = (left_gap + right_gap) / 2
    shoulder_tilt = abs(landmarks[11].y - landmarks[12].y)
    nose_z = landmarks[0].z
    return avg_gap, shoulder_tilt, nose_z

