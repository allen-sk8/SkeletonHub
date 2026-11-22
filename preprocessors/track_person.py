"""Person box detection + tracking (+ appearance Re-ID) -> per-frame boxes shared by all 3D models.

This is the "step 0" of hmr_zoo (and usable alone): video -> RF-DETR person detections -> BoT-SORT tracking
(optionally with OSNet appearance Re-ID) -> choose the main person's track -> repair gaps / collapsed boxes
-> expand the box by a margin so hands and feet are not cut off.

Workflow and defaults come from /home/allen/reid_tracking (Allen's Re-ID tracking trial, read-only here) and
finefs_preproc/scripts/stage1_track.py + stage2_prepare_boxes.py. The original project is untouched.

Must run in the `reidtrack` conda env (rfdetr, trackers, reid, supervision), e.g.

  /home/allen/miniconda3/envs/reidtrack/bin/python preprocessors/track_person.py \
      --input video.mp4 --out-dir outputs/track/video --tracker botsort-reid --device cuda

Options
  --detector  nano|small|medium|base   RF-DETR size (default medium, COCO person class only)
  --tracker   botsort | botsort-reid   without / with OSNet (osnet_x1_0_sportsmot) appearance Re-ID.
              Re-ID reconnects a person who leaves the frame or is occluded; ~costs extra time
              (reid_tracking notes: not needed for a single clean subject; useful with crowds/occlusions).
  --target    auto | largest | point    which track is "the person":
              auto     highest sum of (box area x centrality) over frames  (finefs Stage 2 target rule)
              largest  highest summed box area
              point    track passing nearest to --point X,Y,FRAME
  --margin    box expansion factor around the box centre (default 1.12 = finefs "padded_112")

Outputs in --out-dir
  boxes.json            tight_clean, padded_<margin*100>, padded_112, raw_D (unrepaired target boxes), ids, repaired frames
  boxes_padded_112.txt  'x1 y1 x2 y2' per frame, the format of every SkeletonHub detector's --bbox-file
  tracks.txt            all tracks, MOT rows: frame,id,x1,y1,x2,y2,conf
  detections_<det>.npz  cached raw detections (re-used on re-runs; delete to redo)
  boxes_preview.mp4     (--preview) boxes drawn on the video, check by eye that the right person is tracked
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

ROOT = Path(__file__).resolve().parents[1]
# preprocessors/ has its own `trackers/` package; keep it from shadowing Roboflow's `trackers` when run as a script.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hmr_zoo import boxes as B  # noqa: E402

COCO_PERSON = 1  # RF-DETR keeps original COCO category ids


def detect_all(source: Path, cache: Path, detector: str, det_conf: float, device: str, frames_from_source):
    """RF-DETR person detection for every frame, cached to .npz (same cache format as reid_tracking)."""
    if cache.exists():
        print(f"loaded cached detections: {cache}")
        return np.load(cache)
    import rfdetr

    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
           "base": rfdetr.RFDETRBase}[detector]
    model = cls(device=device)
    fi, bx, sc, n = [], [], [], 0
    t0 = time.perf_counter()
    for _, frame in frames_from_source(str(source)):
        det = model.predict(frame[:, :, ::-1].copy(), threshold=det_conf)  # BGR -> RGB
        det = det[det.class_id == COCO_PERSON]
        fi.append(np.full(len(det), n, dtype=np.int32))
        bx.append(det.xyxy.astype(np.float32) if len(det) else np.zeros((0, 4), np.float32))
        sc.append(det.confidence.astype(np.float32) if len(det) else np.zeros(0, np.float32))
        n += 1
    if n == 0:
        raise RuntimeError(f"no frames read from {source}")
    sec = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, frame_idx=np.concatenate(fi), xyxy=np.concatenate(bx), confidence=np.concatenate(sc),
             num_frames=np.int32(n), det_seconds=np.float32(sec))
    print(f"detected {n} frames in {sec:.1f}s ({n / sec:.1f} fps)")
    return np.load(cache)


def run_tracker(source: Path, dets, tracker, frames_from_source):
    import supervision as sv

    rows = []
    for n, (_, frame) in enumerate(frames_from_source(str(source))):
        if n >= int(dets["num_frames"]):
            break
        m = dets["frame_idx"] == n
        d = sv.Detections(xyxy=dets["xyxy"][m], confidence=dets["confidence"][m],
                          class_id=np.zeros(int(m.sum()), dtype=int))
        d = tracker.update(d, frame=frame)
        d = d[d.tracker_id != -1]
        for tid, b, c in zip(d.tracker_id, d.xyxy, d.confidence):
            rows.append((n, int(tid), *map(float, b), float(c)))
    return np.array(rows).reshape(-1, 7)


def pick_target(tracks: np.ndarray, n: int, w: int, h: int, mode: str, point):
    if mode == "auto":
        return B.target_rule(tracks, n, w, h)
    if mode == "largest":
        area: dict[int, float] = {}
        for _, tid, x1, y1, x2, y2, _ in tracks:
            area[int(tid)] = area.get(int(tid), 0.0) + (x2 - x1) * (y2 - y1)
        best = max(area, key=area.get)
    else:  # point
        px, py, pf = point
        best, best_d = None, 1e18
        for f, tid, x1, y1, x2, y2, _ in tracks:
            if abs(int(f) - pf) > 5:
                continue
            d = np.hypot((x1 + x2) / 2 - px, (y1 + y2) / 2 - py) * (0.1 if x1 <= px <= x2 and y1 <= py <= y2 else 1.0)
            if d < best_d:
                best, best_d = int(tid), d
        if best is None:
            raise SystemExit(f"no track near point {point}")
    box: list = [None] * n
    ids: list = [None] * n
    for f, tid, x1, y1, x2, y2, _ in tracks:
        if int(tid) == best and 0 <= int(f) < n:
            box[int(f)], ids[int(f)] = [float(x1), float(y1), float(x2), float(y2)], best
    return box, ids


def write_preview(mp4: Path, boxes: dict, out: Path, margin_key: str) -> None:
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = min(1.0, 960 / W)
    size = (int(W * sc) // 2 * 2, int(H * sc) // 2 * 2)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                          f"{size[0]}x{size[1]}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-crf", "26",
                          "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    pad, tight = boxes[margin_key], boxes["tight_clean"]
    t = 0
    while t < len(pad):
        ok, fr = cap.read()
        if not ok:
            break
        for b, col in ((pad[t], (60, 245, 70)), (tight[t], (245, 245, 245))):
            cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, max(2, H // 300))
        p.stdin.write(cv2.resize(fr, size).tobytes())
        t += 1
    cap.release()
    p.stdin.close()
    p.wait()


def track_video(video: Path, out_dir: Path, detector="medium", tracker="botsort-reid", target="auto", point=None,
                margin=1.12, det_conf=0.2, device="cuda", gpu_mem_gb=4.0, preview=True) -> dict:
    import torch
    from trackers import BoTSORTTracker, ReIDConfig, frames_from_source

    out_dir.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_mem_gb * 2**30 / total), 0)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    dets = detect_all(video, out_dir / f"detections_{detector}.npz", detector, det_conf, device, frames_from_source)
    n = int(dets["num_frames"])
    common = dict(lost_track_buffer=150, frame_rate=int(round(fps)), high_conf_det_threshold=0.5,
                  track_activation_threshold=0.5, minimum_consecutive_frames=5)
    if tracker == "botsort-reid":
        from reid import ReIDModel

        reid_model = ReIDModel.from_pretrained("osnet_x1_0_sportsmot", device=device)
        trk = BoTSORTTracker(reid_model=reid_model, reid_config=ReIDConfig(
            fusion="adaptive", proximity_threshold=1.0, appearance_threshold=0.09), **common)
    else:
        trk = BoTSORTTracker(**common)
    tracks = run_tracker(video, dets, trk, frames_from_source)
    if len(tracks) == 0:
        raise SystemExit("no person tracked")
    np.savetxt(out_dir / "tracks.txt", tracks, fmt="%d,%d,%.1f,%.1f,%.1f,%.1f,%.3f")
    print(f"{len(np.unique(tracks[:, 1]))} track ids over {n} frames")

    raw, ids = pick_target(tracks, n, W, H, target, point)
    tight, collapsed, gaps = B.repair_boxes(raw, W, H)
    key = f"padded_{int(round(margin * 100))}"
    payload = {"width": W, "height": H, "n_frames": n, "fps": fps, "margin": margin, "detector": detector,
               "tracker": tracker, "target": target, "collapsed_repaired_frames": collapsed,
               "gap_interpolated_frames": gaps, "raw_D": raw, "raw_D_ids": ids,
               "tight_clean": [[round(v, 2) for v in b] for b in tight],
               key: B.expand_boxes(tight, margin, W, H),
               "padded_112": B.expand_boxes(tight, 1.12, W, H), "padded_125": B.expand_boxes(tight, 1.25, W, H)}
    (out_dir / "boxes.json").write_text(json.dumps(payload))
    with (out_dir / "boxes_padded_112.txt").open("w") as f:
        for b in payload["padded_112"]:
            f.write(f"{b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n")
    if preview:
        write_preview(video, payload, out_dir / "boxes_preview.mp4", key)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="person detection + tracking (+Re-ID) -> shared per-frame boxes")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--detector", default="medium", choices=["nano", "small", "medium", "base"])
    ap.add_argument("--tracker", default="botsort-reid", choices=["botsort", "botsort-reid"])
    ap.add_argument("--target", default="auto", choices=["auto", "largest", "point"])
    ap.add_argument("--point", help="X,Y,FRAME for --target point")
    ap.add_argument("--margin", type=float, default=1.12)
    ap.add_argument("--det-conf", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    point = tuple(float(v) for v in a.point.split(",")) if a.point else None
    if a.target == "point" and not point:
        raise SystemExit("--target point needs --point X,Y,FRAME")
    if point:
        point = (point[0], point[1], int(point[2]))
    track_video(Path(a.input), Path(a.out_dir), a.detector, a.tracker, a.target, point, a.margin, a.det_conf,
                a.device, a.gpu_mem_gb, not a.no_preview)
    print(f"boxes -> {a.out_dir}/boxes.json, boxes_padded_112.txt")


if __name__ == "__main__":
    main()
