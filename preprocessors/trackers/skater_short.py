import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from preprocessors.trackers.base_tracker import BaseTracker

class SkaterShortTracker(BaseTracker):
    """
    SkaterShortTracker: Optimizes 2D bounding boxes for short videos (e.g., 5 seconds) 
    using forward-backward dynamic programming.
    
    Ref: /home/allen/datasets/FineFS_5s/scripts/1_yolo_bbox_detect.py
    """
    def __init__(self, weights=None, tracker_cfg=None, device='cuda:0'):
        super().__init__(device)
        self.weights = weights or "/home/allen/yolo/tracking_upgrade/weights/yolo11n.pt"
        if not os.path.exists(self.weights):
            # Fallback to standard yolov8x
            self.weights = "/home/allen/SkeletonHub/external/gvhmr/inputs/checkpoints/yolo/yolov8x.pt"
            
        self.tracker_cfg = tracker_cfg or "/home/allen/yolo/tracking_upgrade/configs/custom_botsort.yaml"
        if not os.path.exists(self.tracker_cfg):
            # Fallback if config does not exist
            self.tracker_cfg = "botsort.yaml"
            
        self.alpha, self.beta, self.gamma = 0.3, 0.7, 0.0
        self.thr_inst = 0.25
        self.short_window = 4
        self.smooth_k = 5

        print(f"⚙️ Initializing SkaterShortTracker with weights: {self.weights}")
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
        ru = 1.0 - self._iou(cur["box"], prev["box"])
        return self.alpha * ra + self.beta * rc + self.gamma * ru

    def _dynamic_sequence(self, frames, w, h, forward=True):
        n = len(frames)
        seq = [None] * n
        if forward and frames[0]:
            seq[0] = max(frames[0], key=lambda d: d["conf"])
        if not forward and frames[-1]:
            seq[-1] = max(frames[-1], key=lambda d: d["conf"])
        rng = range(1, n) if forward else range(n - 2, -1, -1)
        for idx in rng:
            prev = None
            for k in range(1, self.short_window + 1):
                j = idx - k if forward else idx + k
                if 0 <= j < n and seq[j] is not None:
                    prev = seq[j]
                    break
            if prev is None: 
                continue
            best, best_sc = None, 1e9
            for d in frames[idx]:
                sc = self._inst(d, prev, w, h)
                if sc < self.thr_inst and sc < best_sc:
                    best, best_sc = d, sc
            seq[idx] = best
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
        state = {}
        
        # Frame extraction and tracking
        while True:
            ok, frm = cap.read()
            if not ok or frm is None: 
                break
                
            r = self.model.track(
                frm, 
                persist=True, 
                classes=[0], 
                tracker=self.tracker_cfg, 
                conf=0.05, 
                verbose=False
            )[0]
            
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else None
            det = []
            
            for i, bx in enumerate(r.boxes.xyxy.cpu().numpy()):
                tid = int(ids[i]) if ids is not None else i
                conf = float(confs[i]) if confs is not None else 0.0
                x1, y1, x2, y2 = bx
                area = abs((x2 - x1) * (y2 - y1))
                ctr = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                d = {'id': tid, 'box': bx.tolist(), 'conf': conf, 'area': area, 'center': ctr}
                d['inst'] = self._inst(d, state.get(tid), w, h)
                state[tid] = d
                det.append(d)
            frames.append(det)
            
        cap.release()
        N = len(frames)
        if N == 0: 
            raise RuntimeError("❌ Video contains 0 frames")

        # Forward and backward dynamic matching
        seq_f = self._dynamic_sequence(frames, w, h, True)
        seq_b = self._dynamic_sequence(frames, w, h, False)
        final = [(seq_f[i] or seq_b[i] or {}).get("box") for i in range(N)]

        # Linear gap filling
        last = None
        for i in range(N):
            if final[i] is not None: 
                last = i
                continue
            nxt = next((j for j in range(i + 1, N) if final[j] is not None), None)
            if nxt is None or last is None: 
                continue
            b0, b1 = final[last], final[nxt]
            for k in range(i, nxt):
                r = (k - last) / (nxt - last)
                cx = (b0[0] + b0[2]) / 2 * (1 - r) + (b1[0] + b1[2]) / 2 * r
                cy = (b0[1] + b0[3]) / 2 * (1 - r) + (b1[1] + b1[3]) / 2 * r
                w_ = (b0[2] - b0[0]) * (1 - r) + (b1[2] - b1[0]) * r
                h_ = (b0[3] - b0[1]) * (1 - r) + (b1[3] - b1[1]) * r
                final[k] = [cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2]

        # Enlarge and smooth
        enlarged = [self._enlarge_box(b) if b else None for b in final]
        smooth = self._smooth_sequence(enlarged)

        # Convert output to numpy array (N, 4) with fallback for missing frames
        out_boxes = []
        for b in smooth:
            if b is None:
                out_boxes.append([-1.0, -1.0, -1.0, -1.0])
            else:
                out_boxes.append(b)
        return np.array(out_boxes, dtype=np.float32)
