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
  --target    auto | largest | point | lock    which track is "the person":
              auto     highest sum of (box area x centrality) over frames  (finefs Stage 2 target rule)
              largest  highest summed box area
              point    track passing nearest to --point X,Y,FRAME
              lock     for long broadcast videos: seed = the track (>= 1.5 s, person-sized, lower part of the frame) with the
                       largest sum of box area x centrality (the "first track moving >= 3 box heights" rule fails when the
                       camera follows the skater); after a loss
                       or a camera cut the same person is re-attached by OSNet appearance similarity >= 0.6; frames with
                       no such person are marked absent (boxes.json "present": false). Tracks are split at shot cuts
                       (boxes.json "shot_cuts"); "lock" holds the audit trail (seed track, attached tracks + similarities)
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
CUT_DIFF_MIN, CUT_DIFF_RATIO = 25.0, 3.0   # shot cut: mean |gray64x36 diff| above this and above 3x recent median
LOCK_MIN_SEC, LOCK_MIN_MOVE_HEIGHTS = 1.5, 3.0   # (LOCK_MIN_MOVE_HEIGHTS: desktop rule, kept for reference, no longer used)
# Broadcast cameras pan and zoom, so a static spectator also "moves" in the image: a candidate must also look like a person
# on the ice (not a tiny box in the stands): median box height >= LOCK_SEED_MIN_H of the frame height and box centre in the
# lower part of the frame (centre y >= LOCK_MIN_CY of the frame height). Re-attached tracks use the laxer LOCK_ATTACH_MIN_H.
LOCK_SEED_MIN_H, LOCK_ATTACH_MIN_H, LOCK_MIN_CY = 0.20, 0.15, 0.30
LOCK_REID_SIM_DOMINANT = 0.55   # lower bar for a big, central person (box height >= 30% of the frame, centred horizontally)
LOCK_BACKFILL_FRAMES, LOCK_BACKFILL_IOU = 10, 0.3   # BoT-SORT only starts a track after 5 matches: chain detections backwards
LOCK_REID_SIM = 0.6        # appearance similarity to re-attach the same person after a loss / camera cut
LOCK_MIN_FRAMES = 10       # a (sub)track shorter than this is never re-attached
LOCK_MAX_GAP = 12          # frames: shorter missing runs inside one shot are interpolated, longer = person absent


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


def run_tracker(source: Path, dets, tracker, frames_from_source, cuts_out: list | None = None):
    """BoT-SORT over cached detections. If cuts_out is a list, hard shot cuts (broadcast camera switches) found from
    thumbnail differences are appended to it as [frame_index_of_first_frame_of_new_shot, diff] pairs."""
    import supervision as sv

    rows = []
    prev, hist = None, []
    for n, (_, frame) in enumerate(frames_from_source(str(source))):
        if n >= int(dets["num_frames"]):
            break
        if cuts_out is not None:
            g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None:
                d = float(np.abs(g - prev).mean())
                if d > CUT_DIFF_MIN and d > CUT_DIFF_RATIO * max(float(np.median(hist[-15:])) if hist else 0.0, 2.0):
                    cuts_out.append([n, round(d, 1)])
                hist.append(d)
            prev = g
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


def _subtracks(tracks: np.ndarray, cuts: list[int]) -> dict:
    """Split every track at shot cuts: key (track_id, shot_index) -> dict(frames, boxes)."""
    cut_arr = np.asarray(sorted(cuts), dtype=np.int64)
    sub: dict = {}
    for f, tid, x1, y1, x2, y2, _ in tracks:
        key = (int(tid), int(np.searchsorted(cut_arr, int(f), side="right")))
        d = sub.setdefault(key, {"frames": [], "boxes": []})
        d["frames"].append(int(f))
        d["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
    for d in sub.values():
        d["frames"] = np.asarray(d["frames"])
        d["boxes"] = np.asarray(d["boxes"], dtype=np.float64)
    return sub


def _movement_heights(boxes: np.ndarray) -> float:
    """Extent (diagonal) of the box-centre cloud in units of the median box height."""
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    return float(np.hypot(*(c.max(0) - c.min(0))) / max(h, 1.0))


def _embed_samples(video: Path, wanted: dict, reid_model) -> dict:
    """wanted: frame -> list of (key, box). One sequential decode pass; returns key -> (k, D) unit embeddings."""
    import supervision as sv

    out: dict = {}
    if not wanted:
        return out
    last = max(wanted)
    cap = cv2.VideoCapture(str(video))
    n = 0
    while n <= last:
        if n in wanted:
            ok, fr = cap.read()
            if not ok:
                break
            items = wanted[n]
            det = sv.Detections(xyxy=np.asarray([b for _, b in items], dtype=np.float32))
            emb = reid_model.extract_embeddings(det, fr)
            emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
            for (key, _), e in zip(items, emb):
                out.setdefault(key, []).append(e)
        else:
            if not cap.grab():
                break
        n += 1
    cap.release()
    return {k: np.stack(v) for k, v in out.items()}


def _looks_like_skater(boxes: np.ndarray, H: int, min_h: float) -> bool:
    h = float(np.median(boxes[:, 3] - boxes[:, 1])) / H
    cy = float(np.median((boxes[:, 1] + boxes[:, 3]) / 2)) / H
    return h >= min_h and cy >= LOCK_MIN_CY


def _backfill(box: list, ids: list, sub: dict, attached_keys: list, dets, cuts: list[int], n: int) -> int:
    """A new track (e.g. right after a camera cut) only gets an id after a few matched frames, so the first frames of every
    attached sub-track are missing. Chain raw detections backwards from the sub-track's first box (IoU >= LOCK_BACKFILL_IOU),
    at most LOCK_BACKFILL_FRAMES frames and never across a shot cut. Returns number of frames added."""
    if dets is None:
        return 0
    fi, xy = dets["frame_idx"], dets["xyxy"]
    order = np.argsort(fi, kind="stable")
    fi, xy = fi[order], xy[order]
    starts = np.searchsorted(fi, np.arange(n + 1))
    cut_arr = np.asarray(sorted(cuts), dtype=np.int64)
    added = 0
    for key in attached_keys:
        f0 = int(sub[key]["frames"][0])
        cur = np.asarray(sub[key]["boxes"][0], dtype=np.float64)
        shot_start = int(cut_arr[cut_arr <= f0].max()) if np.any(cut_arr <= f0) else 0
        for g in range(f0 - 1, max(shot_start, f0 - LOCK_BACKFILL_FRAMES) - 1, -1):
            if box[g] is not None:
                break
            c = xy[starts[g]:starts[g + 1]].astype(np.float64)
            if len(c) == 0:
                break
            ix = np.maximum(0, np.minimum(c[:, 2], cur[2]) - np.maximum(c[:, 0], cur[0]))
            iy = np.maximum(0, np.minimum(c[:, 3], cur[3]) - np.maximum(c[:, 1], cur[1]))
            inter = ix * iy
            iou = inter / ((c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) + (cur[2] - cur[0]) * (cur[3] - cur[1]) - inter + 1e-9)
            j = int(np.argmax(iou))
            if iou[j] < LOCK_BACKFILL_IOU:
                break
            cur = c[j]
            box[g], ids[g] = [float(v) for v in cur], int(key[0])
            added += 1
    return added


def lock_target(video: Path, tracks: np.ndarray, cuts: list[int], n: int, fps: float, reid_model, H: int = 720, W: int = 1280, dets=None):
    """'Lock onto one skater' for long broadcast videos (rule validated on Allen's lab desktop, re-implemented here).

    1. the seed is the sub-track with the highest sum of box-area x centrality that was seen >= LOCK_MIN_SEC seconds and
       looks like a person on the ice (see below; the desktop's "first track moving >= 3 box heights" fails on broadcast video);
    2. every other sub-track (tracks are split at shot cuts) of >= LOCK_MIN_FRAMES frames that looks like a person on the ice
       is attached to the same person when its OSNet appearance similarity to the seed gallery is >= LOCK_REID_SIM
       (mean pairwise cosine); where several attached tracks overlap in time the most similar one wins;
    3. frames covered by no attached track are 'no person'.
    Returns (box list with None, id list, info dict for the audit trail).
    """
    sub = _subtracks(tracks, cuts)
    min_frames = int(round(LOCK_MIN_SEC * fps))
    # Seed = the (sub)track with the largest sum over frames of box-area x centrality (finefs target rule), among tracks
    # seen >= LOCK_MIN_SEC that look like a person on the ice. The original "first track that moves >= 3 box heights" rule
    # (valid for a fixed camera) fails on broadcast video: the camera follows the skater so the skater hardly moves in the
    # image while spectators "move" with every pan (video 36: it locked onto a spectator).
    half_diag = 0.5 * np.hypot(W, H)
    seed, best = None, 0.0
    for key, d in sub.items():
        b = d["boxes"]
        if len(d["frames"]) < min_frames or not _looks_like_skater(b, H, LOCK_SEED_MIN_H):
            continue
        area = np.maximum(0.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / (W * H)
        dist = np.hypot((b[:, 0] + b[:, 2]) / 2 - W / 2, (b[:, 1] + b[:, 3]) / 2 - H / 2)
        sc = float((area * np.maximum(0.0, 1.0 - dist / half_diag)).sum())
        if sc > best:
            seed, best = key, sc
    info = {"seed": None, "attached": [], "rejected_top": [], "min_frames": min_frames, "seed_rule": "max area*centrality"}
    box: list = [None] * n
    ids: list = [None] * n
    if seed is None:
        return box, ids, info

    def sample(key, k):
        d = sub[key]
        idx = np.unique(np.linspace(0, len(d["frames"]) - 1, k).round().astype(int))
        return [(int(d["frames"][i]), d["boxes"][i].astype(np.float32)) for i in idx]

    wanted: dict = {}
    for f, b in sample(seed, 16):
        wanted.setdefault(f, []).append((("seed",), b))
    cands = [k for k in sub if k != seed and len(sub[k]["frames"]) >= LOCK_MIN_FRAMES
             and _looks_like_skater(sub[k]["boxes"], H, LOCK_ATTACH_MIN_H)]
    for k in cands:
        for f, b in sample(k, 4):
            wanted.setdefault(f, []).append((k, b))
    emb = _embed_samples(video, wanted, reid_model)
    gal = emb[("seed",)]
    sims = {k: float((emb[k] @ gal.T).mean()) for k in cands if k in emb}

    def put(key, tag):
        d = sub[key]
        for f, b in zip(d["frames"], d["boxes"]):
            if 0 <= f < n and box[f] is None:
                box[f], ids[f] = [float(v) for v in b], int(key[0])

    put(seed, "seed")
    attached_keys = [seed]
    info["seed"] = {"id": seed[0], "shot": seed[1], "frames": [int(sub[seed]["frames"][0]), int(sub[seed]["frames"][-1])],
                    "n": int(len(sub[seed]["frames"])), "move_heights": round(_movement_heights(sub[seed]["boxes"]), 1), "score": round(best, 2)}
    for k in sorted(sims, key=lambda k: -sims[k]):
        rec = {"id": k[0], "shot": k[1], "frames": [int(sub[k]["frames"][0]), int(sub[k]["frames"][-1])],
               "n": int(len(sub[k]["frames"])), "sim": round(sims[k], 3)}
        bx = sub[k]["boxes"]
        dominant = (float(np.median(bx[:, 3] - bx[:, 1])) / H >= 0.30
                    and abs(float(np.median((bx[:, 0] + bx[:, 2]) / 2)) - W / 2) <= 0.3 * W)
        rec["dominant"] = bool(dominant)
        if sims[k] >= LOCK_REID_SIM or (dominant and sims[k] >= LOCK_REID_SIM_DOMINANT):
            put(k, "reid")
            attached_keys.append(k)
            info["attached"].append(rec)
        elif len(info["rejected_top"]) < 8:
            info["rejected_top"].append(rec)
    info["backfilled_frames"] = _backfill(box, ids, sub, attached_keys, dets, cuts, n)
    return box, ids, info


def apply_presence(box: list, cuts: list[int], n: int, max_gap: int = LOCK_MAX_GAP):
    """From per-frame boxes with None: present mask. Short gaps inside one shot count as present (boxes interpolated by
    repair_boxes later); longer runs, leading/trailing gaps and gaps across a shot cut are 'no person'."""
    present = np.array([b is not None for b in box], dtype=bool)
    cut_set = np.asarray(sorted(cuts), dtype=np.int64)
    i = 0
    while i < n:
        if present[i]:
            i += 1
            continue
        j = i
        while j < n and not present[j]:
            j += 1
        inner = i > 0 and j < n
        same_shot = inner and not np.any((cut_set > i - 1) & (cut_set <= j))
        if inner and same_shot and (j - i) <= max_gap:
            present[i:j] = True
        i = j
    return present


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
    present = boxes.get("present") or [True] * len(pad)
    t = 0
    while t < len(pad):
        ok, fr = cap.read()
        if not ok:
            break
        for b, col in ((pad[t], (60, 245, 70)), (tight[t], (245, 245, 245))):
            if present[t]:
                cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, max(2, H // 300))
        if not present[t]:
            cv2.putText(fr, "NO PERSON", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, H / 540, (0, 0, 255), 3)
        p.stdin.write(cv2.resize(fr, size).tobytes())
        t += 1
    cap.release()
    p.stdin.close()
    p.wait()


def track_video(video: Path, out_dir: Path, detector="medium", tracker="botsort-reid", target="auto", point=None,
                margin=1.12, det_conf=0.2, device="cuda", gpu_mem_gb=4.0, preview=True, reuse_tracks=False) -> dict:
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

    if reuse_tracks and (out_dir / "tracks.txt").exists() and (out_dir / "boxes.json").exists():
        # re-run only target selection / presence on saved tracks (tracks.txt + shot cuts in the old boxes.json)
        old = json.loads((out_dir / "boxes.json").read_text())
        tracks = np.loadtxt(out_dir / "tracks.txt", delimiter=",").reshape(-1, 7)
        n = int(old["n_frames"])
        cuts_pairs = old.get("shot_cuts", [])
        cuts = [c for c, _ in cuts_pairs]
        track_sec = float(old.get("track_seconds", 0.0))
        dets = {"det_seconds": old.get("det_seconds", 0.0)}
        if (out_dir / f"detections_{detector}.npz").exists():
            z = np.load(out_dir / f"detections_{detector}.npz")
            dets.update(frame_idx=z["frame_idx"], xyxy=z["xyxy"])
        reid_model = None
    else:
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
            reid_model = None
            trk = BoTSORTTracker(**common)
        cuts_pairs: list = []
        t_trk = time.perf_counter()
        tracks = run_tracker(video, dets, trk, frames_from_source, cuts_pairs)
        track_sec = time.perf_counter() - t_trk
        cuts = [c for c, _ in cuts_pairs]
        if len(tracks) == 0:
            raise SystemExit("no person tracked")
        np.savetxt(out_dir / "tracks.txt", tracks, fmt="%d,%d,%.1f,%.1f,%.1f,%.1f,%.3f")
        print(f"{len(np.unique(tracks[:, 1]))} track ids over {n} frames")

    lock_info, t_lock = None, 0.0
    if target == "lock":
        t0 = time.perf_counter()
        if reid_model is None:
            from reid import ReIDModel

            reid_model = ReIDModel.from_pretrained("osnet_x1_0_sportsmot", device=device)
        raw, ids, lock_info = lock_target(video, tracks, cuts, n, fps, reid_model, H, W, dets if not isinstance(dets, dict) or 'frame_idx' in dets else None)
        t_lock = time.perf_counter() - t0
        present = apply_presence(raw, cuts, n)
        if not present.any():
            payload = {"width": W, "height": H, "n_frames": n, "fps": fps, "target": target, "tracker": tracker,
                       "detector": detector, "present": [False] * n, "shot_cuts": cuts_pairs, "lock": lock_info,
                       "no_person": True}
            (out_dir / "boxes.json").write_text(json.dumps(payload))
            print("no person locked (no track >=1.5 s moving >=3 box heights)")
            return payload
    else:
        raw, ids = pick_target(tracks, n, W, H, target, point)
        present = np.ones(n, dtype=bool)
    tight, collapsed, gaps = B.repair_boxes(raw, W, H)
    key = f"padded_{int(round(margin * 100))}"
    payload = {"width": W, "height": H, "n_frames": n, "fps": fps, "margin": margin, "detector": detector,
               "tracker": tracker, "target": target, "collapsed_repaired_frames": collapsed,
               "present": [bool(v) for v in present], "shot_cuts": cuts_pairs, "lock": lock_info,
               "track_seconds": round(track_sec, 2), "lock_seconds": round(t_lock, 2),
               "det_seconds": float(dets["det_seconds"]),
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
    ap.add_argument("--target", default="auto", choices=["auto", "largest", "point", "lock"])
    ap.add_argument("--point", help="X,Y,FRAME for --target point")
    ap.add_argument("--margin", type=float, default=1.12)
    ap.add_argument("--det-conf", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--reuse-tracks", action="store_true", help="skip detection/tracking, redo target selection on saved tracks.txt")
    a = ap.parse_args()
    point = tuple(float(v) for v in a.point.split(",")) if a.point else None
    if a.target == "point" and not point:
        raise SystemExit("--target point needs --point X,Y,FRAME")
    if point:
        point = (point[0], point[1], int(point[2]))
    track_video(Path(a.input), Path(a.out_dir), a.detector, a.tracker, a.target, point, a.margin, a.det_conf,
                a.device, a.gpu_mem_gb, not a.no_preview, a.reuse_tracks)
    print(f"boxes -> {a.out_dir}/boxes.json, boxes_padded_112.txt")


if __name__ == "__main__":
    main()
