# Coursework Tasks 2-5 (Python)

This project runs coursework Tasks 2-5 using OpenCV and writes results under `outputs/`.

## Setup

```bash
python3 -m venv cv_env
source cv_env/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./cv_env/bin/python coursework_tasks.py [options]
```

At the end, the script prints:

```text
Outputs written to: <out_dir>
```

## Main Options

- `--fd-dir`: FD folder for Task 4/5 pair selection.
  - Auto-detect order if omitted: `cv_pictures/FD_uncropped` -> `cv_pictures/FD` -> `cv_pictures/FD_cropped`
- `--fd-no-object-dir`: FD no-object folder (used for Task 3 calibration preference and Task 5 foreground-only outputs).
- `--hg-dir`: HG folder (default `cv_pictures/HG`)
- `--out-dir`: output folder root (default `outputs`)
- `--fd-feature-method`: FD matcher for Task 4/5 (`sift_orb`, `sift`, `orb`, `akaze`; default `akaze`)
- `--pattern-rows`: checkerboard inner-corner rows (default `5`)
- `--pattern-cols`: checkerboard inner-corner cols (default `7`)
- `--min-f-inliers`: Task 5 minimum fundamental inliers gate (default `15`)
- `--min-f-inlier-ratio`: Task 5 minimum fundamental inlier ratio gate (default `0.30`)
- `--rectify-threshold`: threshold passed to `stereoRectifyUncalibrated` (default `5.0`)
- `--no-manual`: skip manual clicking in Task 2
- `--manual-points`: manual correspondences count for Task 2 (default `12`)

## Recommended Commands

Non-interactive run:

```bash
./cv_env/bin/python coursework_tasks.py --no-manual
```

Use uncropped FD for Task 4/5 and aligned no-object FD:

```bash
./cv_env/bin/python coursework_tasks.py \
  --fd-dir cv_pictures/FD_uncropped \
  --fd-no-object-dir cv_pictures/FD_uncropped_no_object \
  --hg-dir cv_pictures/HG \
  --no-manual
```

If you want to test another dataset root (example: `cv_pictures_2`):

```bash
./cv_env/bin/python coursework_tasks.py \
  --fd-dir cv_pictures_2/FD_uncropped \
  --fd-no-object-dir cv_pictures_2/FD_uncropped_no_object \
  --hg-dir cv_pictures_2/HG \
  --no-manual
```

Relax Task 5 gate for debugging weak FD pairs:

```bash
./cv_env/bin/python coursework_tasks.py \
  --no-manual \
  --min-f-inliers 12 \
  --min-f-inlier-ratio 0.20
```

## Task Inputs/Behavior

- Task 2:
  - Uses HG images only.
  - Runs two automatic matching pipelines on the same selected HG pair.
  - Optional manual correspondence mode.

- Task 3:
  - Uses FD no-object images when at least 3 are available.
  - Otherwise falls back to FD + HG + no-object combined.
  - Writes calibration logs and distortion illustration.

- Task 4:
  - Estimates homography from HG pair.
  - Estimates fundamental matrix from FD pair.
  - Uses calibration undistortion for FD matching if Task 3 succeeds.

- Task 5:
  - Uses the FD pair selected by Task 4.
  - Applies pre-rectification fundamental support gate (`--min-f-*`).
  - Generates rectified pair, disparity, and relative depth.
  - Foreground-only outputs are produced only if FD no-object indices align with selected FD pair.

## Output Structure

- `outputs/task2/`
  - `automatic_matches.jpg`
  - `automatic_matches_<method>.jpg`
  - `manual_matches.jpg` (manual mode only)
  - `metrics.txt`

- `outputs/task3/`
  - `chessboard_detection_log.txt`
  - `camera_parameters.txt`
  - `distortion_illustration.jpg`
  - `calibration_failed.txt` (only if calibration fails)

- `outputs/task4/`
  - `homography_projected_keypoints.jpg`
  - `epipolar_lines_in_image2.jpg`
  - `epipole_vanishing_horizon.jpg` (if vanishing-point estimation succeeds)
  - `matrices_and_metrics.txt`

- `outputs/task5/`
  - Success:
    - `rectified_pair_with_epipolar_lines.jpg`
    - `disparity_map.jpg`
    - `relative_depth_map.jpg`
    - `quality_metrics.txt`
    - optional:
      - `foreground_mask_from_fd_no_object.jpg`
      - `disparity_map_foreground_only.jpg`
      - `relative_depth_map_foreground_only.jpg`
      - `foreground_metrics.txt`
  - Failure:
    - `rectification_failed.txt`

## Notes

- Checkerboard settings are **inner corners** (not printed square count).
- Task 5 depth is relative depth (`1/disparity`), not metric depth.
- Do not mix cropped and uncropped FD images in one folder for final results.
