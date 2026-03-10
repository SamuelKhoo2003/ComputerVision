# Coursework Tasks 2-5 (Python)

This repository runs coursework Tasks 2-5 using OpenCV.

## Setup

```bash
python3 -m venv cv_env
source cv_env/bin/activate
pip install -r requirements.txt
```

## CLI Usage

```bash
./cv_env/bin/python coursework_tasks.py [options]
```

Options:
- `--fd-dir`: FD image folder.
  - Default auto-detect order: `cv_pictures/FD_uncropped` -> `cv_pictures/FD` -> `cv_pictures/FD_cropped`
- `--fd-no-object-dir`: no-object FD folder (optional override)
- `--hg-dir`: HG image folder (default `cv_pictures/HG`)
- `--out-dir`: output root folder (default `outputs`)
- `--pattern-rows`: checkerboard inner-corner rows for calibration (default `5`)
- `--pattern-cols`: checkerboard inner-corner cols for calibration (default `7`)
- `--no-manual`: disable manual clicks in Task 2
- `--manual-points`: number of manual correspondences for Task 2 (default `12`)

## Recommended Runs

Non-interactive:

```bash
./cv_env/bin/python coursework_tasks.py --no-manual
```

Explicit split (Task 4/5 from uncropped FD):

```bash
./cv_env/bin/python coursework_tasks.py \
  --fd-dir cv_pictures/FD_uncropped \
  --fd-no-object-dir cv_pictures/FD_no_object \
  --hg-dir cv_pictures/HG \
  --no-manual
```

## What Each Task Uses

- Task 2:
  - Uses `HG` images only.
  - Runs two automatic pipelines on the same selected HG pair:
    - primary: `SIFT` (fallback `AKAZE` if SIFT unavailable)
    - secondary: `ORB`
  - Optional manual correspondence comparison.

- Task 3:
  - Prefers `FD_no_object` images if at least 3 are available.
  - Falls back to combined FD + HG + no-object if needed.
  - Performs camera calibration and writes intrinsics/distortion.

- Task 4:
  - Homography on HG.
  - Fundamental matrix on FD.
  - If calibration is available from Task 3, FD images are undistorted before FD matching/model fitting.

- Task 5:
  - Uses the FD pair selected in Task 4.
  - Applies a pre-rectification quality gate on Task 4 fundamental support:
    - minimum inliers: `35`
    - minimum inlier ratio: `0.30`
  - If gate fails, writes `rectification_failed.txt` and skips disparity/depth generation.
  - If no-object FD pair is index-aligned, also computes foreground-only outputs.

## Output Structure

All outputs are written under `outputs/`.

- `outputs/task2/`
  - `automatic_matches.jpg`
  - `automatic_matches_sift.jpg` (or `automatic_matches_akaze.jpg`)
  - `automatic_matches_orb.jpg`
  - `manual_matches.jpg` (manual mode only)
  - `metrics.txt`

- `outputs/task3/`
  - `chessboard_detection_log.txt`
  - `camera_parameters.txt`
  - `distortion_illustration.jpg`
  - `calibration_failed.txt` (only when calibration is not possible)

- `outputs/task4/`
  - `homography_projected_keypoints.jpg`
  - `epipolar_lines_in_image2.jpg`
  - `epipole_vanishing_horizon.jpg` (if VP/horizon estimation succeeds)
  - `matrices_and_metrics.txt`

- `outputs/task5/`
  - Success path:
    - `rectified_pair_with_epipolar_lines.jpg`
    - `disparity_map.jpg`
    - `relative_depth_map.jpg`
    - `quality_metrics.txt`
    - optional foreground files when no-object alignment exists:
      - `foreground_mask_from_fd_no_object.jpg`
      - `disparity_map_foreground_only.jpg`
      - `relative_depth_map_foreground_only.jpg`
      - `foreground_metrics.txt`
  - Failure path:
    - `rectification_failed.txt`

## Notes

- Checkerboard pattern uses **inner corners**, not number of printed squares.
- Task 5 depth is relative (`1/disparity`), not metric depth.
- If cropped/uncropped FD images are mixed in one folder, Task 4/5 pair quality usually degrades. Prefer separate runs for each set.
