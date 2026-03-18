# Smart Posture Tracker

Smart Posture Tracker is a Windows desktop app that uses MediaPipe Pose to estimate basic posture metrics in real time and provides immediate feedback via an on-screen warning overlay.

## What it does

- Reads your webcam pose stream in real time.
- Computes three posture metrics:
  - `gap` (shoulder-to-head vertical distance; lower usually means slouching)
  - `tilt` (left vs right shoulder height difference; higher usually means side lean)
  - `z` (head position depth; lower usually means forward-head posture)
- Shows live metrics in the app and highlights values that are outside the active threshold range.
- Supports calibration to personalize thresholds.
- Optionally shows a warning overlay (icon + optional text + optional sound).

## How thresholds work

The evaluator considers posture **bad** when any of these are true:

- `z_depth < thresholds.z`
- `gap < thresholds.gap`
- `tilt > thresholds.tilt`

Calibration produces new `gap`, `z`, and `tilt` thresholds and uses them immediately.

## Metrics coloring (home page)

On the home page, the live metrics are color-coded:

- Red values: outside the current effective thresholds
- Black values: inside the current effective thresholds

## Configuration

User settings are saved to:

- `%APPDATA%/SmartPostureTracker/config.json`

This includes:

- `use_manual_thresholds` and threshold values
- `is_calibrated` plus calibrated thresholds
- camera index
- preview toggle
- overlay settings (enabled, position, sound, text)

## Quick start (development)

1. Install dependencies

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Run the app

   ```powershell
   python main.py
   ```



