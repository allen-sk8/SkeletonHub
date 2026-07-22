import os
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from preprocessors.trackers.base_tracker import BaseTracker

class SkaterLongTracker(BaseTracker):
    """
    SkaterLongTracker: Optimizes 2D bounding boxes for longer video sequences (e.g., skater routines)
    using stable anchor points, target re-acquisition, background filtering, and gap limits.
    
    Ref: /home/Henry/Skeleton-Toolkit/scripts/skeleton_extraction/1_yolo_bbox_detect.py
    """
    def __init__(self, weights=None, device='cuda:0'):
        super().__init__(device)
        self.weights = weights or "/home/allen/yolo/tracking_upgrade/weights/yolo11n.pt"
        if not os.path.exists(self.weights):
            # Fallback to standard yolov8x
            self.weights = "/home/allen/SkeletonHub/external/gvhmr/inputs/checkpoints/yolo/yolov8x.pt"

        self.alpha, self.beta = 0.3, 0.7
        self.thr_inst = 0.25
        self.short_window = 4
        self.area_ratio_limit = 3.0
        self.stable_conf_thr = 0.5
        self.stable_area_ratio_limit = 4.0
        self.stable_update_ratio = 2.0
        self.recover_conf = 0.4
        self.recover_patience = 5
        self.recover_bg_window = 10
        self.dom_conf = 0.6
        self.dom_ratio = 2.5
        self.dom_patience = 5
        self.smooth_k = 5
        self.max_gap_fill = 12

        print(f"⚙️ Initializing SkaterLongTracker with weights: {self.weights}")
        try:
            self.model = YOLO(self.weights)
        except Exception as e:
            print(f"⚠️ Failed to load YOLO weights '{self.weights}' due to: {e}. Falling back to standard YOLOv8x.")
            self.weights = "/home/allen/SkeletonHub/external/gvhmr/inputs/checkpoints/yolo/yolov8x.pt"
            self.model = YOLO(self.weights)
        self.model.to(self.device)

    def _iou(self, a, b):
        if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]: 
            return 0.0
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter) if areaA + areaB - inter else 0.0

    def _inst(self, cur, prev, w, h):
        if prev is None: 
            return 1e3
        ra = abs(np.log((cur["area"] + 1e-6) / (prev["area"] + 1e-6)))
        rc = np.linalg.norm(cur["center"] - prev["center"]) / np.hypot(w, h)
        return self.alpha * ra + self.beta * rc

    def _dynamic_sequence(self, frames, w, h, forward=True):
        n = len(frames)
        seq = [None] * n
        if forward and frames[0]:
            seq[0] = max(frames[0], key=lambda d: d["conf"])
        if not forward and frames[-1]:
            seq[-1] = max(frames[-1], key=lambda d: d["conf"])
            
        stable_ref = seq[0] if forward else seq[-1]
        dom_count = 0
        rec_cand, rec_count = None, 0
        bg_hist = deque(maxlen=self.recover_bg_window)
        rng = range(1, n) if forward else range(n - 2, -1, -1)
        
        for idx in rng:
            prev = None
            for k in range(1, self.short_window + 1):
                j = idx - k if forward else idx + k
                if 0 <= j < n and seq[j] is not None:
                    prev = seq[j]
                    break
            if prev is None:
                # Re-anchor from last validated position
                ref = None
                for j in range(idx - 1, -1, -1) if forward else range(idx + 1, n):
                    if seq[j] is not None:
                        ref = seq[j]
                        break
                j = idx - 1 if forward else idx + 1
                if 0 <= j < n and frames[j]:
                    if ref is not None:
                        cands = [d for d in frames[j]
                                 if max(d["area"], ref["area"]) / max(min(d["area"], ref["area"]), 1e-6) <= self.area_ratio_limit]
                        if cands:
                            prev = min(cands, key=lambda d: self._inst(d, ref, w, h))
                    else:
                        prev = max(frames[j], key=lambda d: d["conf"])
            
            if prev is not None:
                best, best_sc = None, 1e9
                for d in frames[idx]:
                    area_ratio = max(d["area"], prev["area"]) / max(min(d["area"], prev["area"]), 1e-6)
                    if area_ratio > self.area_ratio_limit: 
                        continue
                    if stable_ref is not None:
                        stable_ratio = max(d["area"], stable_ref["area"]) / max(min(d["area"], stable_ref["area"]), 1e-6)
                        if stable_ratio > self.stable_area_ratio_limit: 
                            continue
                    sc = self._inst(d, prev, w, h)
                    if sc < self.thr_inst and sc < best_sc:
                        best, best_sc = d, sc
                seq[idx] = best
                
                # Update stable anchor reference
                if best is not None and best["conf"] >= self.stable_conf_thr:
                    ratio = max(best["area"], stable_ref["area"]) / max(min(best["area"], stable_ref["area"]), 1e-6) if stable_ref is not None else 1.0
                    if ratio <= self.stable_update_ratio:
                        stable_ref = best
                
                # Check dominant competitor for re-anchoring
                cur = best if best is not None else prev
                big = max(frames[idx], key=lambda d: d["area"]) if frames[idx] else None
                if (big is not None and big["conf"] >= self.dom_conf
                        and big["area"] >= self.dom_ratio * cur["area"]):
                    dom_count += 1
                    if dom_count >= self.dom_patience:
                        seq[idx] = big
                        stable_ref = big
                        dom_count = 0
                else:
                    dom_count = 0
            
            # Loss of tracking recovery logic
            if seq[idx] is None:
                cand = max((d for d in frames[idx]
                          if d["conf"] >= self.recover_conf
                          and not any(self._inst(d, b, w, h) < self.thr_inst for bg in bg_hist for b in bg)),
                         key=lambda d: d["conf"], default=None)
                if cand is not None and rec_cand is not None and self._inst(cand, rec_cand, w, h) < self.thr_inst:
                    rec_count += 1
                else:
                    rec_count = 1 if cand is not None else 0
                rec_cand = cand
                if rec_count >= self.recover_patience:
                    seq[idx] = cand
                    stable_ref = cand
            
            # History recording for recovery
            if seq[idx] is not None:
                tgt = seq[idx]
                bg_hist.append([d for d in frames[idx]
                                if d is not tgt and self._iou(d["box"], tgt["box"]) < 0.5])
                rec_cand, rec_count = None, 0
                
        return seq

    def _enlarge_box(self, b, s=1.1):
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        w = (b[2] - b[0]) * s
        h = (b[3] - b[1]) * s
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def _smooth_sequence(self, boxes):
        n = len(boxes)
        out = [None] * n
        r = self.smooth_k // 2
        for i, b in enumerate(boxes):
            if b is None: 
                continue
            xs, ys, ars = [], [], []
            for j in range(max(0, i - r), min(n, i + r + 1)):
                if boxes[j] is None: 
                    continue
                bx = boxes[j]
                xs.append((bx[0] + bx[2]) / 2)
                ys.append((bx[1] + bx[3]) / 2)
                ars.append((bx[2] - bx[0]) * (bx[3] - bx[1]))
            if not xs: 
                continue
            cx, cy, ar = np.mean(xs), np.mean(ys), np.mean(ars)
            ratio = (b[2] - b[0]) / (b[3] - b[1] + 1e-6)
            h = np.sqrt(ar / ratio)
            w = h * ratio
            out[i] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        return out

    def track(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video: {video_path}")
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ok, frm = cap.read()
            if not ok or frm is None: 
                break
                
            r = self.model.predict(frm, classes=[0], conf=0.05, verbose=False)[0]
            det = []
            for bx, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = bx
                area = abs((x2 - x1) * (y2 - y1))
                ctr = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                det.append({'box': bx.tolist(), 'conf': float(cf), 'area': area, 'center': ctr})
            frames.append(det)
            
        cap.release()
        N = len(frames)
        if N == 0: 
            raise RuntimeError("❌ Video contains 0 frames")

        seq_f = self._dynamic_sequence(frames, w, h, True)
        seq_b = self._dynamic_sequence(frames, w, h, False)
        final = [(seq_f[i] or seq_b[i] or {}).get("box") for i in range(N)]

        # Linear gap filling (up to max_gap_fill)
        last = None
        for i in range(N):
            if final[i] is not None: 
                last = i
                continue
            nxt = next((j for j in range(i + 1, N) if final[j] is not None), None)
            if nxt is None or last is None: 
                continue
            if nxt - last - 1 > self.max_gap_fill: 
                continue
            b0, b1 = final[last], final[nxt]
            for k in range(i, nxt):
                r = (k - last) / (nxt - last)
                cx = (b0[0] + b0[2]) / 2 * (1 - r) + (b1[0] + b1[2]) / 2 * r
                cy = (b0[1] + b0[3]) / 2 * (1 - r) + (b1[1] + b1[3]) / 2 * r
                w_ = (b0[2] - b0[0]) * (1 - r) + (b1[2] - b1[0]) * r
                h_ = (b0[3] - b0[1]) * (1 - r) + (b1[3] - b1[1]) * r
                final[k] = [cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2]

        enlarged = [self._enlarge_box(b) if b else None for b in final]
        smooth = self._smooth_sequence(enlarged)

        out_boxes = []
        for b in smooth:
            if b is None:
                out_boxes.append([-1.0, -1.0, -1.0, -1.0])
            else:
                out_boxes.append(b)
        return np.array(out_boxes, dtype=np.float32)
