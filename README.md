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

## Build a Windows EXE (PyInstaller)

This repo includes a helper script for consistent builds:

```powershell
.\build_exe.ps1
```

The expected output is:

- `dist/SmartPostureTracker.exe`

If you prefer to run PyInstaller directly:

```powershell
python -m PyInstaller --noconfirm --clean SmartPostureTracker.spec
```

## GitHub Actions (CI/CD) for automatic EXE builds

This repo includes a workflow at:

- `.github/workflows/build-windows-exe.yml`

How it works:

- When you push a tag that matches `v*`, GitHub Actions builds the EXE on Windows.
- The workflow uploads the `.exe` to the workflow artifact and (for `v*` tags) attaches it to the corresponding GitHub Release.

### Recommended release flow

1. Commit your changes and push them to GitHub.
2. Create a tag (for example):

   ```powershell
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. Ensure a GitHub Release exists for that tag (or create one once in the GitHub UI).
4. Download `SmartPostureTracker.exe` from the Release Assets (or Actions artifacts).

## Troubleshooting

- If the EXE fails to run due to MediaPipe missing modules, rebuild with the included spec (or ensure you use the spec/CI workflow rather than a minimal PyInstaller command).

