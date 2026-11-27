"""2D body keypoints (COCO 17) every few frames, for enroll_identity.py (front / side / back view and upright check).

To be merged later with the 2D model comparison in experiments/pose2d_bench/ (not yet committed). Only used to tell
front / side / back here, so the model was not chosen for accuracy.

Runs YOLO11m-pose on the whole frame (imgsz 1920 so 40-px people are still found) every --stride frames.
Must run in an env with ultralytics (desktop: onlinehmr), e.g.

  ~/miniconda3/envs/onlinehmr/bin/python preprocessors/pose2d_frames.py --video V.mp4 --out OUT/pose2d.npz \
      --weights /path/to/yolo11m-pose.pt

Output npz: frame (N,), box (N,4), kpt (N,17,3) = x, y, confidence in original video pixels; stride; frames; wall_s.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="yolo11m-pose.pt", help="ultralytics downloads it if only a name is given")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.2)
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.video)
    F, Bx, K, n = [], [], [], 0
    t0 = time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n % a.stride == 0:
            r = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False, half=True)[0]
            if r.keypoints is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy(); k = r.keypoints.data.cpu().numpy()
                F.append(np.full(len(b), n)); Bx.append(b); K.append(k)
        n += 1
    wall = time.time() - t0
    np.savez(a.out, frame=np.concatenate(F).astype(np.int32) if F else np.zeros(0, np.int32),
             box=np.concatenate(Bx).astype(np.float32) if Bx else np.zeros((0, 4), np.float32),
             kpt=np.concatenate(K).astype(np.float32) if K else np.zeros((0, 17, 3), np.float32),
             stride=a.stride, frames=n, wall_s=wall)
    print(f"pose2d {a.video}: {n} frames, {wall:.1f}s")


if __name__ == "__main__":
    main()
