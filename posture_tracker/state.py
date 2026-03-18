from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PostureState:
    # Frame/UI state
    latest_annotated_frame: Optional[np.ndarray] = None
    posture_status: str = "Initializing..."
    current_metrics: Dict[str, float] = field(
        default_factory=lambda: {"gap": 0, "tilt": 0, "z_depth": 0}
    )

    # Calibration & detection state
    is_calibrated: bool = False
    calibration_data: List[Tuple[float, float]] = field(default_factory=list)
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {"gap": 0.20, "z": -1.10}
    )  # Default fallbacks

    bad_posture_start_time: Optional[float] = None
    alert_active: bool = False

