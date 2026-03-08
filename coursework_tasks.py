#!/usr/bin/env python3
import argparse
import glob
import os
import re
from dataclasses import dataclass
from itertools import combinations

# Avoid OpenCV OpenCL cache write failures on restricted temp dirs.
os.environ.setdefault("OPENCV_OPENCL_CACHE_ENABLE", "0")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Disable OpenCL immediately after import so no kernels are queued before main().
if hasattr(cv2, "ocl"):
    cv2.ocl.setUseOpenCL(False)


@dataclass
class MatchResult:
    pts1: np.ndarray
    pts2: np.ndarray
    kp1: list
    kp2: list
    matches: list
    vis: np.ndarray


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def natural_sorted_jpgs(folder: str):
    if not os.path.isdir(folder):
        return []

    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    def key(p: str):
        name = os.path.basename(p)
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

    return sorted(set(paths), key=key)


def resolve_first_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def read_img(path: str):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_and_match(img1, img2, max_features=4000, ratio=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Prefer SIFT for stability. Fallback to ORB where SIFT is unavailable.
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("Not enough keypoints/descriptors found.")

    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError("Not enough good matches after ratio test.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, good[:300], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return MatchResult(pts1=pts1, pts2=pts2, kp1=kp1, kp2=kp2, matches=good, vis=vis)


def select_best_pair(paths, estimator="homography"):
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images to select a pair.")

    best = None
    best_score = -1.0
    failures = 0

    for i, j in combinations(range(len(paths)), 2):
        img1 = read_img(paths[i])
        img2 = read_img(paths[j])
        try:
            m = detect_and_match(img1, img2)
        except RuntimeError:
            failures += 1
            continue

        if estimator == "homography":
            M, mask = cv2.findHomography(m.pts1, m.pts2, cv2.RANSAC, 3.0)
        elif estimator == "fundamental":
            M, mask = cv2.findFundamentalMat(m.pts1, m.pts2, cv2.FM_RANSAC, 1.5, 0.99)
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        if M is None or mask is None:
            failures += 1
            continue

        inliers = int(np.sum(mask))
        inlier_ratio = float(mask.mean())
        score = inliers + 1000.0 * inlier_ratio
        if score > best_score:
            best_score = score
            best = {
                "i": i,
                "j": j,
                "img1": img1,
                "img2": img2,
                "match": m,
                "model": M,
                "mask": mask,
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
            }

    if best is None:
        raise RuntimeError(
            f"Could not find a valid {estimator} pair across {len(paths)} images "
            f"(failures: {failures})."
        )
    return best


def manual_correspondences(img1, img2, n_points=12):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(bgr_to_rgb(img1))
    axes[0].set_title(f"Image 1: click {n_points} points")
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(img2))
    axes[1].set_title(f"Image 2: click same {n_points} points, same order")
    axes[1].axis("off")

    plt.tight_layout()
    print(f"Click {n_points} points in LEFT image, press Enter.")
    plt.sca(axes[0])
    p1 = plt.ginput(n=n_points, timeout=0)

    print(f"Click corresponding {n_points} points in RIGHT image, press Enter.")
    plt.sca(axes[1])
    p2 = plt.ginput(n=n_points, timeout=0)
    plt.close(fig)

    if len(p1) != n_points or len(p2) != n_points:
        raise RuntimeError("Manual point collection incomplete.")

    return np.float32(p1), np.float32(p2)


def save_image(path, img_bgr):
    cv2.imwrite(path, img_bgr)


def hstack_with_padding(img1, img2, pad_value=0):
    """Stack images horizontally by padding the shorter one at the bottom."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 == h2:
        return np.hstack([img1, img2])

    max_h = max(h1, h2)
    out1 = np.full((max_h, w1, 3), pad_value, dtype=img1.dtype)
    out2 = np.full((max_h, w2, 3), pad_value, dtype=img2.dtype)
    out1[:h1, :w1] = img1
    out2[:h2, :w2] = img2
    return np.hstack([out1, out2])


def reproj_error_homography(H, pts1, pts2):
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    proj = (H @ pts1_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    e = np.linalg.norm(proj - pts2, axis=1)
    return e


def run_task2(hg_paths, out_dir, do_manual=True, manual_points=12):
    t2_dir = os.path.join(out_dir, "task2")
    ensure_dir(t2_dir)

    best = select_best_pair(hg_paths, estimator="homography")
    img1 = best["img1"]
    img2 = best["img2"]
    auto = best["match"]
    H_auto = best["model"]
    mask_auto = best["mask"]
    inlier_ratio_auto = float(mask_auto.mean())
    err_auto = reproj_error_homography(H_auto, auto.pts1, auto.pts2)

    save_image(os.path.join(t2_dir, "automatic_matches.jpg"), auto.vis)

    metrics = {
        "auto_pair_indices": (int(best["i"]), int(best["j"])),
        "auto_num_matches": int(len(auto.matches)),
        "auto_inlier_ratio": inlier_ratio_auto,
        "auto_reproj_error_mean": float(np.mean(err_auto)),
        "auto_reproj_error_median": float(np.median(err_auto)),
    }

    if do_manual:
        p1, p2 = manual_correspondences(img1, img2, n_points=manual_points)
        H_man, _ = cv2.findHomography(p1, p2, 0)
        if H_man is None:
            raise RuntimeError("Task 2 manual: homography estimation failed.")
        err_man = reproj_error_homography(H_man, p1, p2)

        # Draw manual correspondences
        vis = hstack_with_padding(img1.copy(), img2.copy())
        w = img1.shape[1]
        for a, b in zip(p1, p2):
            c = tuple(np.random.randint(0, 255, size=3).tolist())
            pa = (int(a[0]), int(a[1]))
            pb = (int(b[0] + w), int(b[1]))
            cv2.circle(vis, pa, 5, c, -1)
            cv2.circle(vis, pb, 5, c, -1)
            cv2.line(vis, pa, pb, c, 2)
        save_image(os.path.join(t2_dir, "manual_matches.jpg"), vis)

        metrics.update({
            "manual_num_matches": int(len(p1)),
            "manual_reproj_error_mean": float(np.mean(err_man)),
            "manual_reproj_error_median": float(np.median(err_man)),
        })

    with open(os.path.join(t2_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return metrics


def detect_chessboard_points(img_paths, rows, cols):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp_swapped = np.zeros((rows * cols, 3), np.float32)
    objp_swapped[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    used = []
    diagnostics = []

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)

    for p in img_paths:
        img = read_img(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        method = "findChessboardCorners"
        used_swapped = False

        if not ok and hasattr(cv2, "findChessboardCornersSB"):
            ok_sb, corners_sb = cv2.findChessboardCornersSB(gray, (cols, rows), None)
            if ok_sb:
                ok = True
                corners = corners_sb
                method = "findChessboardCornersSB"

        swapped_ok = False
        if not ok:
            sw_ok, sw_corners = cv2.findChessboardCorners(gray, (rows, cols), flags)
            swapped_ok = bool(sw_ok)
            if not sw_ok and hasattr(cv2, "findChessboardCornersSB"):
                sw_ok_sb, sw_corners_sb = cv2.findChessboardCornersSB(gray, (rows, cols), None)
                if sw_ok_sb:
                    sw_ok = True
                    sw_corners = sw_corners_sb
            if sw_ok:
                ok = True
                corners = sw_corners
                used_swapped = True
                method = "findChessboardCorners(swapped)"

        diagnostics.append({
            "path": p,
            "shape": (int(gray.shape[0]), int(gray.shape[1])),
            "ok": bool(ok),
            "method": method if ok else "none",
            "swapped_pattern_ok": swapped_ok,
            "used_swapped_pattern": used_swapped,
        })

        if not ok:
            continue
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        objpoints.append(objp_swapped.copy() if used_swapped else objp.copy())
        imgpoints.append(corners2)
        used.append(p)

    return objpoints, imgpoints, used, diagnostics


def draw_distortion_comparison(img, K, dist):
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    und = cv2.undistort(img, K, dist, None, newK)

    vis = np.hstack([img, und])
    for y in np.linspace(40, h - 40, 8).astype(int):
        cv2.line(vis, (20, y), (w - 20, y), (0, 255, 255), 1)
        cv2.line(vis, (w + 20, y), (2 * w - 20, y), (0, 255, 255), 1)
    return vis


def run_task3(all_paths, out_dir, pattern_rows=5, pattern_cols=7):
    t3_dir = os.path.join(out_dir, "task3")
    ensure_dir(t3_dir)

    objpoints, imgpoints, used, diagnostics = detect_chessboard_points(
        all_paths, pattern_rows, pattern_cols
    )

    with open(os.path.join(t3_dir, "chessboard_detection_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"requested_pattern_rows: {pattern_rows}\n")
        f.write(f"requested_pattern_cols: {pattern_cols}\n")
        f.write(f"images_considered: {len(all_paths)}\n")
        f.write(f"images_detected: {len(used)}\n\n")
        for d in diagnostics:
            f.write(
                f"ok={d['ok']}, method={d['method']}, swapped_pattern_ok={d['swapped_pattern_ok']}, "
                f"used_swapped_pattern={d.get('used_swapped_pattern', False)}, "
                f"shape(h,w)={d['shape']}, path={d['path']}\n"
            )

    if len(objpoints) < 3:
        with open(os.path.join(t3_dir, "calibration_failed.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Chessboard detection failed on most images. "
                "Re-run with correct --pattern-rows/--pattern-cols or clearer grid images.\n"
            )
            f.write(
                f"Detected checkerboards in {len(used)} out of {len(all_paths)} images.\n"
                "See chessboard_detection_log.txt for per-image diagnostics.\n"
            )
        return None

    sample = read_img(used[0])
    h, w = sample.shape[:2]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    with open(os.path.join(t3_dir, "camera_parameters.txt"), "w", encoding="utf-8") as f:
        f.write(f"images_used: {len(used)}\n")
        f.write(f"rms_reprojection_error: {rms}\n")
        f.write("K:\n")
        f.write(str(K) + "\n")
        f.write("dist_coeffs:\n")
        f.write(str(dist.ravel()) + "\n")

    vis = draw_distortion_comparison(sample, K, dist)
    save_image(os.path.join(t3_dir, "distortion_illustration.jpg"), vis)

    return {
        "K": K,
        "dist": dist,
        "rms": rms,
    }


def draw_projected_points(img1, img2, pts1, pts2, H):
    vis = hstack_with_padding(img1.copy(), img2.copy())
    w = img1.shape[1]

    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    proj = (H @ pts1_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]

    for p2, pp in zip(pts2, proj):
        c = tuple(np.random.randint(0, 255, size=3).tolist())
        q = (int(pp[0] + w), int(pp[1]))
        r = (int(p2[0] + w), int(p2[1]))
        cv2.circle(vis, q, 4, c, -1)
        cv2.circle(vis, r, 4, (255, 255, 255), 1)
        cv2.line(vis, q, r, c, 1)

    return vis


def compute_epipoles(F):
    # Right epipole: F e = 0, Left epipole: F^T e' = 0
    _, _, vt = np.linalg.svd(F)
    e_right = vt[-1]
    e_right = e_right / e_right[2]

    _, _, vt2 = np.linalg.svd(F.T)
    e_left = vt2[-1]
    e_left = e_left / e_left[2]
    return e_left, e_right


def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    vis2 = img2.copy()

    h, w = img2.shape[:2]
    for r, p2 in zip(lines2, pts2):
        a, b, c = r
        if abs(b) < 1e-8:
            continue
        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.line(vis2, (x0, y0), (x1, y1), color, 1)
        cv2.circle(vis2, (int(p2[0]), int(p2[1])), 4, color, -1)

    return vis2


def estimate_vanishing_points_and_horizon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    if lines is None or len(lines) < 20:
        return None, None, None

    lines = lines[:, 0, :]

    lengths = np.linalg.norm(lines[:, :2] - lines[:, 2:], axis=1)
    keep = lengths > np.percentile(lengths, 60)
    lines = lines[keep]
    if len(lines) < 10:
        return None, None, None

    angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
    feats = np.float32(np.stack([np.cos(2 * angles), np.sin(2 * angles)], axis=1))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    _, labels, _ = cv2.kmeans(feats, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    def cluster_intersection(c):
        ls = lines[labels.ravel() == c]
        if len(ls) < 2:
            return None
        A = []
        b = []
        for x1, y1, x2, y2 in ls:
            a = y1 - y2
            bb = x2 - x1
            cc = x1 * y2 - x2 * y1
            A.append([a, bb])
            b.append([-cc])
        A = np.float64(A)
        b = np.float64(b)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return np.array([x[0, 0], x[1, 0], 1.0])

    vp1 = cluster_intersection(0)
    vp2 = cluster_intersection(1)
    if vp1 is None or vp2 is None:
        return None, None, None

    horizon = np.cross(vp1, vp2)
    if abs(horizon[2]) > 1e-8:
        horizon = horizon / horizon[2]

    return vp1, vp2, horizon


def draw_vps_horizon(img, vp1, vp2, horizon):
    vis = img.copy()
    h, w = img.shape[:2]

    for vp, color in [(vp1, (0, 0, 255)), (vp2, (0, 255, 0))]:
        x, y = int(vp[0]), int(vp[1])
        cv2.circle(vis, (np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)), 10, color, 2)

    a, b, c = horizon
    if abs(b) > 1e-8:
        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)
        cv2.line(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)

    return vis


def estimate_outlier_tolerance(pts1, pts2, max_trials=30):
    # Ground truth from robust estimate on original correspondences
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return None

    in1 = pts1[mask.ravel() == 1]
    in2 = pts2[mask.ravel() == 1]
    if len(in1) < 8:
        return None

    ratios = np.linspace(0.0, 0.9, 10)
    survivals = []

    xmn, xmx = float(np.min(pts1[:, 0])), float(np.max(pts1[:, 0]))
    ymn, ymx = float(np.min(pts1[:, 1])), float(np.max(pts1[:, 1]))
    umn, umx = float(np.min(pts2[:, 0])), float(np.max(pts2[:, 0]))
    vmn, vmx = float(np.min(pts2[:, 1])), float(np.max(pts2[:, 1]))

    for r in ratios:
        recalls = []
        for _ in range(max_trials):
            n_in = len(in1)
            n_out = int((r / (1.0 - r + 1e-8)) * n_in)
            o1 = np.column_stack([
                np.random.uniform(xmn, xmx, n_out),
                np.random.uniform(ymn, ymx, n_out),
            ]).astype(np.float32)
            o2 = np.column_stack([
                np.random.uniform(umn, umx, n_out),
                np.random.uniform(vmn, vmx, n_out),
            ]).astype(np.float32)

            p1 = np.vstack([in1, o1])
            p2 = np.vstack([in2, o2])

            H_r, m_r = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
            if H_r is None or m_r is None:
                recalls.append(0.0)
                continue
            recovered_inliers = int(np.sum(m_r[:n_in]))
            recalls.append(recovered_inliers / max(n_in, 1))

        survivals.append(float(np.median(recalls)))

    tol = 0.0
    for rr, rec in zip(ratios, survivals):
        if rec >= 0.5:
            tol = float(rr)

    return ratios, survivals, tol


def run_task4(hg_paths, fd_paths, out_dir):
    t4_dir = os.path.join(out_dir, "task4")
    ensure_dir(t4_dir)

    # 4.1 Homography on HG
    best_hg = select_best_pair(hg_paths, estimator="homography")
    hg1 = best_hg["img1"]
    hg2 = best_hg["img2"]
    m_hg = best_hg["match"]
    H = best_hg["model"]
    mH = best_hg["mask"]

    in1 = m_hg.pts1[mH.ravel() == 1]
    in2 = m_hg.pts2[mH.ravel() == 1]
    vis_proj = draw_projected_points(hg1, hg2, in1, in2, H)
    save_image(os.path.join(t4_dir, "homography_projected_keypoints.jpg"), vis_proj)

    # 4.2 Fundamental matrix on FD
    best_fd = select_best_pair(fd_paths, estimator="fundamental")
    fd1 = best_fd["img1"]
    fd2 = best_fd["img2"]
    m_fd = best_fd["match"]
    F = best_fd["model"]
    mF = best_fd["mask"]

    f1 = m_fd.pts1[mF.ravel() == 1]
    f2 = m_fd.pts2[mF.ravel() == 1]

    epi_img2 = draw_epipolar_lines(fd1, fd2, f1, f2, F)
    save_image(os.path.join(t4_dir, "epipolar_lines_in_image2.jpg"), epi_img2)

    ep_l, ep_r = compute_epipoles(F)

    vp1, vp2, horizon = estimate_vanishing_points_and_horizon(fd1)
    if vp1 is not None:
        vh = draw_vps_horizon(fd1, vp1, vp2, horizon)
        x, y = int(np.clip(ep_l[0], 0, fd1.shape[1] - 1)), int(np.clip(ep_l[1], 0, fd1.shape[0] - 1))
        cv2.circle(vh, (x, y), 9, (255, 255, 0), 2)
        save_image(os.path.join(t4_dir, "epipole_vanishing_horizon.jpg"), vh)

    # 4.3 Outlier tolerance
    outlier = estimate_outlier_tolerance(m_hg.pts1, m_hg.pts2)

    with open(os.path.join(t4_dir, "matrices_and_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"HG pair indices (task4.1): {best_hg['i']}, {best_hg['j']}\n")
        f.write(f"FD pair indices (task4.2/5): {best_fd['i']}, {best_fd['j']}\n\n")
        f.write("Homography H:\n")
        f.write(str(H) + "\n\n")
        f.write("Fundamental F:\n")
        f.write(str(F) + "\n\n")
        f.write(f"Homography inliers: {len(in1)}/{len(m_hg.pts1)}\n")
        f.write(f"Fundamental inliers: {len(f1)}/{len(m_fd.pts1)}\n")
        f.write(f"Left epipole: {ep_l}\n")
        f.write(f"Right epipole: {ep_r}\n")
        if outlier is not None:
            ratios, survivals, tol = outlier
            f.write(f"Estimated outlier tolerance (median recall>=0.5): {tol}\n")
            for rr, ss in zip(ratios, survivals):
                f.write(f"outlier_ratio={rr:.2f}, median_recall={ss:.3f}\n")

    return {
        "F": F,
        "fd_inlier_pts1": f1,
        "fd_inlier_pts2": f2,
        "fd1": fd1,
        "fd2": fd2,
        "fd_i": int(best_fd["i"]),
        "fd_j": int(best_fd["j"]),
    }


def draw_horizontal_guides(img, n=15):
    vis = img.copy()
    h, w = vis.shape[:2]
    ys = np.linspace(20, h - 20, n).astype(int)
    for y in ys:
        cv2.line(vis, (0, y), (w, y), (0, 255, 255), 1)
    return vis


def run_task5(fd1, fd2, F, pts1, pts2, out_dir, fd1_bg=None, fd2_bg=None):
    t5_dir = os.path.join(out_dir, "task5")
    ensure_dir(t5_dir)

    h, w = fd1.shape[:2]

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1.reshape(-1, 2), pts2.reshape(-1, 2), F, imgSize=(w, h)
    )
    if not ok:
        with open(os.path.join(t5_dir, "rectification_failed.txt"), "w", encoding="utf-8") as f:
            f.write("stereoRectifyUncalibrated failed. Try another FD pair.\n")
        return

    r1 = cv2.warpPerspective(fd1, H1, (w, h))
    r2 = cv2.warpPerspective(fd2, H2, (w, h))

    vis_pair = np.hstack([draw_horizontal_guides(r1), draw_horizontal_guides(r2)])
    save_image(os.path.join(t5_dir, "rectified_pair_with_epipolar_lines.jpg"), vis_pair)

    g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 10,
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = sgbm.compute(g1, g2).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    # Relative depth (unknown metric scale without known baseline/focal pair)
    rel_depth = 1.0 / (disp + 1e-6)
    rel_depth[np.isnan(rel_depth)] = 0
    rel_depth = cv2.normalize(rel_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    disp_vis = np.nan_to_num(disp, nan=0.0)
    disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    depth_color = cv2.applyColorMap(rel_depth, cv2.COLORMAP_TURBO)

    save_image(os.path.join(t5_dir, "disparity_map.jpg"), disp_color)
    save_image(os.path.join(t5_dir, "relative_depth_map.jpg"), depth_color)

    if fd1_bg is not None and fd2_bg is not None:
        rb1 = cv2.warpPerspective(fd1_bg, H1, (w, h))
        rb2 = cv2.warpPerspective(fd2_bg, H2, (w, h))

        d1 = cv2.cvtColor(cv2.absdiff(r1, rb1), cv2.COLOR_BGR2GRAY)
        d2 = cv2.cvtColor(cv2.absdiff(r2, rb2), cv2.COLOR_BGR2GRAY)
        _, m1 = cv2.threshold(d1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, m2 = cv2.threshold(d2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, kernel)
        m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernel)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, kernel)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.bitwise_and(m1, m2)
        save_image(os.path.join(t5_dir, "foreground_mask_from_fd_no_object.jpg"), fg_mask)

        fg_disp = np.nan_to_num(disp.copy(), nan=0.0)
        fg_disp[fg_mask == 0] = 0
        fg_disp_vis = cv2.normalize(fg_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fg_disp_color = cv2.applyColorMap(fg_disp_vis, cv2.COLORMAP_TURBO)
        save_image(os.path.join(t5_dir, "disparity_map_foreground_only.jpg"), fg_disp_color)

        fg_depth = rel_depth.copy()
        fg_depth[fg_mask == 0] = 0
        fg_depth_color = cv2.applyColorMap(fg_depth, cv2.COLORMAP_TURBO)
        save_image(os.path.join(t5_dir, "relative_depth_map_foreground_only.jpg"), fg_depth_color)

        with open(os.path.join(t5_dir, "foreground_metrics.txt"), "w", encoding="utf-8") as f:
            fg_pixels = int(np.count_nonzero(fg_mask))
            total_pixels = int(fg_mask.size)
            f.write(f"foreground_pixels: {fg_pixels}\n")
            f.write(f"foreground_fraction: {fg_pixels / max(total_pixels, 1):.6f}\n")



def main():
    # Keep processing on CPU for reproducibility and to avoid OpenCL cache errors.
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    parser = argparse.ArgumentParser(description="Computer Vision coursework Tasks 2-5")
    parser.add_argument("--fd-dir", default="cv_pictures/FD")
    parser.add_argument("--fd-no-object-dir", default="")
    parser.add_argument("--hg-dir", default="cv_pictures/HG")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--pattern-rows", type=int, default=5, help="Chessboard inner corners rows")
    parser.add_argument("--pattern-cols", type=int, default=7, help="Chessboard inner corners cols")
    parser.add_argument("--no-manual", action="store_true", help="Disable manual clicking for task 2")
    parser.add_argument("--manual-points", type=int, default=12)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    fd_paths = natural_sorted_jpgs(args.fd_dir)
    hg_paths = natural_sorted_jpgs(args.hg_dir)
    fd_no_object_dir = args.fd_no_object_dir or resolve_first_existing_dir(
        [
            "cv_pictures/FD_no_banana",
            "cv_pictures/FD_no_object",
            "cv_pictures/FD_without_object",
            "cv_pictures/FD without object",
        ]
    )
    fd_no_object_paths = natural_sorted_jpgs(fd_no_object_dir) if fd_no_object_dir else []

    if len(fd_paths) < 2 or len(hg_paths) < 2:
        raise RuntimeError("Need at least 2 images in each of FD and HG folders.")

    if fd_no_object_dir:
        print(f"Using no-object FD folder: {fd_no_object_dir} ({len(fd_no_object_paths)} images)")
    else:
        print("No FD no-object folder found. Task 5 foreground-only outputs will be skipped.")

    print("Running Task 2...")
    run_task2(
        hg_paths,
        args.out_dir,
        do_manual=not args.no_manual,
        manual_points=args.manual_points,
    )

    print("Running Task 3...")
    if len(fd_no_object_paths) >= 3:
        calib_paths = fd_no_object_paths
        print(f"Task 3 calibration source: FD no-object only ({len(calib_paths)} images)")
    else:
        calib_paths = fd_paths + hg_paths + fd_no_object_paths
        print(
            "Task 3 calibration source: fallback to FD + HG + FD no-object "
            f"({len(calib_paths)} images total)"
        )
    run_task3(calib_paths, args.out_dir, args.pattern_rows, args.pattern_cols)

    print("Running Task 4...")
    t4 = run_task4(hg_paths, fd_paths, args.out_dir)

    print("Running Task 5...")
    fd1_bg = None
    fd2_bg = None
    if len(fd_no_object_paths) > max(t4["fd_i"], t4["fd_j"]):
        fd1_bg = read_img(fd_no_object_paths[t4["fd_i"]])
        fd2_bg = read_img(fd_no_object_paths[t4["fd_j"]])
    elif fd_no_object_paths:
        print(
            "FD no-object images exist but counts/order do not align with chosen FD pair; "
            "skipping foreground-only depth outputs."
        )

    run_task5(
        fd1=t4["fd1"],
        fd2=t4["fd2"],
        F=t4["F"],
        pts1=t4["fd_inlier_pts1"],
        pts2=t4["fd_inlier_pts2"],
        out_dir=args.out_dir,
        fd1_bg=fd1_bg,
        fd2_bg=fd2_bg,
    )

    print("Done. Outputs written to:", args.out_dir)


if __name__ == "__main__":
    main()
