from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .main_window import MainWindow


def run() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Smart Posture Tracker")
    win = MainWindow()
    win.show()
    return app.exec()

