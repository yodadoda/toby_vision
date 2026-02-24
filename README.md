# TOBY Vision

A computer-vision subsystem for TOBY that detects objects with YOLO, locks onto people, and computes tracking error relative to the camera center.

## What this codebase does
- Discovers available cameras and lets you select one at runtime.
- Captures frames at a configured FPS.
- Runs YOLO object detection per frame.
- Draws detection boxes:
  - Person boxes in red.
  - Non-person boxes in black.
  - Locked person box in thicker red.
- Locks onto a person target using a face-biased point (top-third center of the person box).
- Computes tracking error (`dx`, `dy`, and Euclidean distance in pixels) from frame center.
- Applies a deadzone so minor movement near center is treated as centered.
- Displays everything in a popup window and prints tracking telemetry in the terminal.

## Project structure
- `eyes.py`: thin entrypoint.
- `toby_vision/config.py`: centralized tunable settings.
- `toby_vision/camera.py`: camera discovery/selection/opening.
- `toby_vision/detection.py`: YOLO wrapper and normalized detections.
- `toby_vision/tracking.py`: person-lock and error/deadzone math.
- `toby_vision/overlay.py`: visualization (boxes, labels, deadzone, centers).
- `toby_vision/app.py`: main runtime loop.
- `tests/test_tracking.py`: unit tests for tracking logic.

## Setup (from clone to run)
1. Clone and enter the repo:
```bash
git clone <your-repo-url>
cd toby_vision
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Place model weights:
- Put `yolo26n.pt` in the repository root (same directory as `eyes.py`),
- or change `model_path` in `toby_vision/config.py`.

5. Run the app:
```bash
python3 eyes.py
```

6. At runtime:
- Select a camera index in the terminal.
- A popup window opens with live tracking output.
- Press `q` in the popup to exit.

## Tracking behavior
- Target candidate class: `person` only.
- Person lock point:
  - `cx = (x1 + x2) / 2`
  - `cy = y1 + 0.33 * (y2 - y1)`
- Error math:
  - `dx = target_x - frame_center_x`
  - `dy = target_y - frame_center_y`
  - `distance = sqrt(dx^2 + dy^2)`
- Deadzone defaults:
  - `deadzone_x_px = 30`
  - `deadzone_y_px = 30`

## Config tuning
Edit `VisionConfig` in `toby_vision/config.py`:
- `frame_rate`
- `confidence_threshold`
- `deadzone_x_px`, `deadzone_y_px`
- `lock_persistence_radius_px`
- `camera_index`, `max_camera_scan_index`
- `mirror_camera`
- drawing colors/thickness values

## Running tests
```bash
python3 -m unittest discover -s tests -v
```

## Troubleshooting
- `No cameras detected`:
  - Ensure webcam is connected and not used by another app.
  - On macOS, allow Terminal/IDE camera permissions.
- Model load error:
  - Confirm `yolo26n.pt` exists in repo root or update `model_path`.
- Low FPS:
  - Lower `frame_rate` or `confidence_threshold`, or use a smaller model.
