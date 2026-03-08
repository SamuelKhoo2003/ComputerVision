# Coursework Tasks 2-5 (Python)

This script implements Tasks 2, 3, 4, 5 using images in:
- `cv_pictures/FD`
- `cv_pictures/FD_no_banana` (FD without object)
- `cv_pictures/HG`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Interactive mode (includes manual correspondences in Task 2):

```bash
python coursework_tasks.py
```

Non-interactive mode (skip manual clicking):

```bash
python coursework_tasks.py --no-manual
```

If calibration fails, provide correct chessboard pattern size (inner corners):

```bash
python coursework_tasks.py --pattern-rows 6 --pattern-cols 9

If your "FD without object" folder has a custom path/name:

```bash
python coursework_tasks.py --fd-no-object-dir "cv_pictures/FD without object"
```
```

## Outputs

All artifacts are saved under `outputs/`:

- `outputs/task2/`: automatic vs manual correspondences + comparison metrics
- `outputs/task3/`: camera intrinsics/distortion + undistortion illustration
- `outputs/task4/`: homography/fundamental results, epipolar lines, epipoles, vanishing points/horizon, outlier tolerance
- `outputs/task5/`: rectified stereo pair and disparity/relative depth maps
  - plus foreground-only disparity/depth maps when `FD_no_banana` is available
