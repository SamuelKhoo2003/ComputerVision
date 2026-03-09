# Coursework Tasks 2-5 (Python)

This project runs coursework Tasks 2-5 using images from:
- `cv_pictures/FD`
- `cv_pictures/HG`
- `cv_pictures/FD_no_object` (or `FD_no_banana` / equivalent name) for no-object calibration and foreground masking

## Setup

```bash
python3 -m venv cv_env
source cv_env/bin/activate
pip install -r requirements.txt
```

## Run

Interactive mode (includes manual clicks for Task 2):

```bash
./cv_env/bin/python coursework_tasks.py
```

Non-interactive mode:

```bash
./cv_env/bin/python coursework_tasks.py --no-manual
```

If needed, set chessboard inner-corner pattern explicitly (default is `5x7`):

```bash
./cv_env/bin/python coursework_tasks.py --pattern-rows 5 --pattern-cols 7
```

If your no-object folder uses a custom name/path:

```bash
./cv_env/bin/python coursework_tasks.py --fd-no-object-dir "cv_pictures/FD without object"
```

## Output Structure

All artifacts are written under `outputs/`.

- `outputs/task2/`
  - `automatic_matches.jpg`
  - `automatic_matches_sift.jpg` / `automatic_matches_orb.jpg` (or `akaze` if SIFT unavailable)
  - `manual_matches.jpg` (only when interactive manual mode is used)
  - `metrics.txt`
- `outputs/task3/`
  - `chessboard_detection_log.txt`
  - `camera_parameters.txt`
  - `distortion_illustration.jpg`
  - `calibration_failed.txt` (only if insufficient detections)
- `outputs/task4/`
  - `homography_projected_keypoints.jpg`
  - `epipolar_lines_in_image2.jpg`
  - `epipole_vanishing_horizon.jpg` (when VP/horizon estimation succeeds)
  - `matrices_and_metrics.txt`
- `outputs/task5/`
  - `rectified_pair_with_epipolar_lines.jpg`
  - `disparity_map.jpg`
  - `relative_depth_map.jpg`
  - `quality_metrics.txt`
  - `foreground_mask_from_fd_no_object.jpg`, foreground-only depth/disparity, and `foreground_metrics.txt` (when aligned no-object pair exists)
  - `rectification_failed.txt` (if rectification overlap is too small or stereo ROI is invalid)

## Notes on Interpretation

- Task 3 pattern size is **inner corners**, not checkerboard squares.
- When `FD_no_object` exists, Task 4/5 FD pair selection is automatically limited to indices shared with `FD` so foreground-only depth can be computed.
- In Task 5 rectification checks, horizontal epipolar lines should align across both images by `y` level; they do not need to pass exactly through checkerboard intersections.
- Task 5 depth is relative (`1/disparity`), not metric depth, because baseline and absolute scale are unknown.
