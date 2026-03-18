from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6 import QtGui


def bgr_to_qimage(frame_bgr: np.ndarray) -> Optional[QtGui.QImage]:
    if frame_bgr is None:
        return None
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return None
    h, w, _ = frame_bgr.shape
    rgb = frame_bgr[:, :, ::-1].copy()
    bytes_per_line = 3 * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()

