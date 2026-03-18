from __future__ import annotations

import os
import sys
from pathlib import Path


def resource_path(relative_path: str) -> str:
    """
    Return an absolute path to a resource.

    Works in development and when packaged by PyInstaller (sys._MEIPASS).
    """
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path:
        return str(Path(base_path) / relative_path)
    return str(Path(__file__).resolve().parents[2] / relative_path)


def appdata_dir(app_name: str = "SmartPostureTracker") -> Path:
    base = os.environ.get("APPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Roaming")
    return Path(base) / app_name

