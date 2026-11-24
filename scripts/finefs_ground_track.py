"""Horizontal travel on the ice for FineFS v2 (prototype, CPU only).

GVHMR's world trajectory barely moves (median ~0.4 m/s; real skating 3-7 m/s): the broadcast camera follows the skater,
SimpleVO underestimates the pans by ~4x and ignores zoom, and GVHMR's learned motion prior expects steps, not gliding.
Here the skater's motion over the ice is measured from the image instead:
  1. background motion between consecutive frames: ORB features at 0.5x outside the (enlarged) skater box, RANSAC
     homography. The broadcast camera rotates and zooms in place, so one homography describes the whole background.
  2. skater's motion relative to the scene, in pixels: where the skater's ground point (box bottom centre) is now, minus
     where the background homography moves last frame's ground point to.
  3. pixels to metres with GVHMR's camera-space depth of the pelvis z (focal-length errors cancel: z and pixels both use
     GVHMR's assumed focal f): sideways = dx * z / f; towards / away from the camera = dy * z / (f * sin(elevation)),
     elevation = how far the camera looks down: fixed ELEV_DEG (GVHMR's own estimate, ~5 deg, is far too flat).
  4. the two horizontal components are put into GVHMR's world frame along the camera's horizontal right / forward axes
     of that frame, so the travel direction stays consistent with GVHMR's body orientation (forward vs backward skating).
  5. per segment: velocities median-filtered and smoothed, flights (jump frames) interpolated for the vertical pixel
     component (a jump moves the box up, not away), then integrated to a path. Vertical comes from finefs_jump_height.py.

  python scripts/finefs_ground_track.py --id 0 [--frames 400-800]     # prints speeds; library: ground_velocity()
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.spatial.transform import Rotation as Rot

D = Path("/home/allen/datasets/FineFS")


def background_homographies(video, boxes, f0, f1, scale=0.5):
    """H[t] maps frame t-1 to frame t (full-resolution pixels), for t in f0+1..f1-1; NaN where it failed."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    S = np.diag([scale, scale, 1.0])
    Hs = np.full((f1 - f0, 3, 3), np.nan)
    inl = np.zeros(f1 - f0)
    prev = None
    for k in range(f1 - f0):
        ok, fr = cap.read()
        if not ok:
            break
        gr = cv2.cvtColor(cv2.resize(fr, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        m = np.full(gr.shape, 255, np.uint8)
        b = boxes[f0 + k]
        if b is not None:
            x1, y1, x2, y2 = [int(c * scale) for c in b]
            w, h = x2 - x1, y2 - y1
            m[max(0, y1 - h // 4):y2 + h // 4, max(0, x1 - w // 2):x2 + w // 2] = 0
        kp, des = orb.detectAndCompute(gr, m)
        if prev is not None and des is not None and prev[1] is not None and len(kp) > 20:
            mt = bf.match(prev[1], des)
            if len(mt) > 30:
                a = np.float32([prev[0][x.queryIdx].pt for x in mt])
                c = np.float32([kp[x.trainIdx].pt for x in mt])
                H, mask = cv2.findHomography(a, c, cv2.RANSAC, 2.0)
                if H is not None and mask.sum() > 25:
                    Hs[k] = np.linalg.inv(S) @ H @ S
                    inl[k] = mask.mean()
        prev = (kp, des)
    cap.release()
    return Hs, inl


ELEV_DEG = 20.0        # assumed camera elevation (main broadcast camera high in the stands); GVHMR's own estimate is ~5 deg,
                       # which makes towards/away motion 4x too fast (spikes of 40-60 m/s)
MAX_SPEED = 12.0       # m/s; faster than any skater: scaled down to this


def ground_velocity(g, bj, f0, f1, flights=(), video=None, elev_deg=ELEV_DEG):
    """Per-frame horizontal velocity (T,2) [world x, world z] in m/s for frames f0..f1-1 (one GVHMR segment)."""
    fps = float(g["fps"])
    f = float(g["K"][0, 0])
    boxes = [bj["tight_clean"][t] if bj["present"][t] else None for t in range(len(bj["present"]))]
    Hs, inl = background_homographies(video or D / "video" / f"{bj.get('id', '')}.mp4", boxes, f0, f1)
    n = f1 - f0
    gp = np.full((n, 2), np.nan)                                 # ground point: bottom centre of the skater box
    for k in range(n):
        b = boxes[f0 + k]
        if b is not None:
            gp[k] = [(b[0] + b[2]) / 2, b[3]]
    d = np.full((n, 2), np.nan)                                  # skater motion relative to the scene, px/frame
    for k in range(1, n):
        if np.isfinite(Hs[k]).all() and np.isfinite(gp[k]).all() and np.isfinite(gp[k - 1]).all():
            q = Hs[k] @ np.array([gp[k - 1, 0], gp[k - 1, 1], 1.0])
            d[k] = gp[k] - q[:2] / q[2]
    air = np.zeros(n, bool)
    for a, b, *_ in flights:
        air[max(0, a - f0 - 2):max(0, b - f0 + 2)] = True        # the box moves up in the air: not depth motion
    d[air, 1] = np.nan
    for c in range(2):                                           # fill gaps, remove spikes
        ok = np.isfinite(d[:, c])
        d[:, c] = np.interp(np.arange(n), np.where(ok)[0], d[ok, c]) if ok.sum() > 2 else 0
        d[:, c] = uniform_filter1d(median_filter(d[:, c], 5, mode="nearest"), 5)
    z = median_filter(g["joints24_cam"][f0:f1, 0, 2].astype(float), 25, mode="nearest")
    # camera axes in GVHMR world: R_c<-w = R_cam_body @ R_world_body^T
    Rw = Rot.from_rotvec(g["smpl_world_global_orient"][f0:f1]).as_matrix()
    Rc = Rot.from_rotvec(g["smpl_cam_global_orient"][f0:f1]).as_matrix()
    Rcw = Rc @ np.transpose(Rw, (0, 2, 1))
    right = Rcw[:, 0, :]                                         # camera x axis in world (rows of R_c<-w)
    fwd = Rcw[:, 2, :]                                           # optical axis in world
    elev_gv = np.arcsin(np.clip(-fwd[:, 1], -1, 1))             # GVHMR's estimate, kept for reference only
    elev = np.full(n, np.radians(elev_deg)) if elev_deg else median_filter(np.clip(elev_gv, np.radians(5), np.radians(60)), 51, mode="nearest")
    rh = right[:, [0, 2]] / np.linalg.norm(right[:, [0, 2]], axis=1, keepdims=True)
    fh = fwd[:, [0, 2]] / np.linalg.norm(fwd[:, [0, 2]], axis=1, keepdims=True)
    side = d[:, 0] * z / f                                       # m/frame to the right in the image
    away = -d[:, 1] * z / (f * np.sin(elev))                     # up in the image = away from the camera
    v = (side[:, None] * rh + away[:, None] * fh) * fps
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    v = v * np.minimum(1.0, MAX_SPEED / np.maximum(sp, 1e-9))
    return v, {"inlier_frac": inl, "elev_deg": np.degrees(elev), "elev_gvhmr_deg": np.degrees(elev_gv), "z": z, "d_px": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--frames", default=None, help="a-b; default: the longest segment")
    a = ap.parse_args()
    g = np.load(D / "3_gvhmr" / a.id / "gvhmr_fast.npz")
    bj = json.loads((D / "1_tracking" / a.id / "boxes.json").read_text()); bj["id"] = a.id
    if a.frames:
        f0, f1 = map(int, a.frames.split("-"))
    else:
        s = g["segments"]; i = int(np.argmax(s[:, 1] - s[:, 0])); f0, f1 = int(s[i, 0]), int(s[i, 1])
    v, info = ground_velocity(g, bj, f0, f1)
    sp = np.linalg.norm(v, axis=1)
    old = np.linalg.norm(np.diff(g["joints24_world"][f0:f1, 0][:, [0, 2]], axis=0), axis=1) * float(g["fps"])
    print(f"frames {f0}-{f1}: speed median {np.median(sp):.2f} m/s, 90th pct {np.percentile(sp, 90):.2f}, max {sp.max():.1f} "
          f"| GVHMR median {np.median(old):.2f}, 90th {np.percentile(old, 90):.2f} | camera looks down {np.median(info['elev_deg']):.0f} deg "
          f"(GVHMR says {np.median(info['elev_gvhmr_deg']):.0f}), "
          f"background inliers {np.median(info['inlier_frac']):.2f}, path length {sp.sum() / float(g['fps']):.0f} m")


if __name__ == "__main__":
    main()
