"""Offline 'lock onto one skater' for whole broadcast videos (new file; preprocessors/track_person.py is untouched and its
--target lock stays available for comparison).

Why: the online lock (track_person.py --target lock) uses BoT-SORT tracks, which only get an id after 5 consecutive matched
detections, and it chains raw detections backwards for at most 10 frames. So after every camera cut the first 3 to 9
frames have no person. Here the whole video is available, so every shot is handled with all of its detections, looking both
forwards and backwards:

  1. shots = frame ranges between hard cuts (cuts as found by track_person.py, read from its boxes.json);
  2. tracklets per shot = BoT-SORT tracks split at cuts, each extended BACKWARDS AND FORWARDS through the cached raw
     detections (detections_<det>.npz) by IoU chaining up to the shot boundary (IoU >= EXT_IOU, height ratio within
     EXT_HRATIO, at most EXT_MAX_MISS missing frames in a row; stops when the chained detection belongs to another
     BoT-SORT track). Detections that no tracklet covers are linked into extra tracklets (greedy IoU, same rules), so a
     shot where BoT-SORT never started a track still has candidates;
  3. appearance: the seed (same rule as the online lock: largest sum of box area x centrality among tracks >= 1.5 s
     that look like a person on the ice) gives a gallery of SEED_SAMPLES OSNet embeddings; every tracklet is embedded at
     TRK_SAMPLES frames spread over its WHOLE length (not 4 frames of the BoT-SORT part only). Score = mean pairwise cosine
     to the gallery (same scale and thresholds as the online lock: 0.6, 0.55 for a big central person).
     Second pass: tracklets accepted with score >= GALLERY_ADD_SIM add their samples to the gallery (whole-video
     appearance average), every tracklet is re-scored, final score = max(seed-only score, expanded-gallery score);
  4. per frame: BoT-SORT parts of accepted tracklets first (most similar first), then their extensions, then extra
     tracklets, each only into frames still empty; short gaps inside a shot are interpolated
     (track_person.apply_presence, <= 12 frames), everything else is 'no person'.

Output boxes.json has the same keys as track_person.py (present, shot_cuts, tight_clean, padded_112, ...), so
detectors/gvhmr_fast_detector.py reads it unchanged; "lock" holds the audit trail and "target" is "lock_offline".

Run in the reidtrack env (needs reid / OSNet), CPU or GPU:
  python preprocessors/track_person_offline.py --video V.mp4 --track-dir outputs/finefs_full_test/36 --out-dir OUT
    --track-dir must hold boxes.json (for shot cuts), tracks.txt and detections_medium.npz from track_person.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hmr_zoo import boxes as B  # noqa: E402
from preprocessors.track_person import (LOCK_ATTACH_MIN_H, LOCK_MIN_FRAMES, LOCK_MIN_SEC, LOCK_REID_SIM,  # noqa: E402
                                        LOCK_REID_SIM_DOMINANT, LOCK_SEED_MIN_H, _embed_samples, _looks_like_skater,
                                        _subtracks, apply_presence)

EXT_IOU = 0.3          # IoU to chain a raw detection onto a tracklet end
EXT_HRATIO = 1.5       # box height may change at most by this factor between chained boxes
EXT_MAX_MISS = 2       # frames without a matching detection before a chain stops
OWN_IOU = 0.7          # a detection with IoU >= this to another BoT-SORT track's box belongs to that track
COVER_IOU = 0.5        # a detection overlapping a tracklet box this much is 'covered' (not used for extra tracklets)
EXTRA_MIN_CONF = 0.3   # extra tracklets are built from detections at least this confident
SEED_SAMPLES, TRK_SAMPLES = 16, 8
GALLERY_ADD_SIM = 0.70
GALLERY_MAX = 96


def _iou(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.maximum(0, np.minimum(c[:, 2], b[2]) - np.maximum(c[:, 0], b[0]))
    iy = np.maximum(0, np.minimum(c[:, 3], b[3]) - np.maximum(c[:, 1], b[1]))
    inter = ix * iy
    return inter / ((c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Dets:
    def __init__(self, z, n):
        fi = np.asarray(z["frame_idx"])
        order = np.argsort(fi, kind="stable")
        self.fi, self.xy, self.cf = fi[order], np.asarray(z["xyxy"], np.float64)[order], np.asarray(z["confidence"])[order]
        self.st = np.searchsorted(self.fi, np.arange(n + 1))

    def at(self, f):
        return self.xy[self.st[f]:self.st[f + 1]], self.cf[self.st[f]:self.st[f + 1]]


def _chain(cur, f_from, f_to, step, dets: Dets, trk_by_frame: dict, own_id):
    """Chain detections from box `cur` at frame f_from-step... towards f_to (inclusive). Returns list of (frame, box)."""
    out, miss = [], 0
    g = f_from + step
    while (g <= f_to if step > 0 else g >= f_to):
        c, _ = dets.at(g)
        ok = False
        if len(c):
            iou = _iou(c, cur)
            j = int(np.argmax(iou))
            hr = (c[j, 3] - c[j, 1]) / max(cur[3] - cur[1], 1.0)
            if iou[j] >= EXT_IOU and 1 / EXT_HRATIO <= hr <= EXT_HRATIO:
                others = [b for tid, b in trk_by_frame.get(g, []) if tid != own_id]
                if others and _iou(np.asarray(others), c[j]).max() >= OWN_IOU:
                    break                                  # that detection is another tracked person
                cur = c[j]
                out.append((g, cur.copy()))
                ok, miss = True, 0
        if not ok:
            miss += 1
            if miss > EXT_MAX_MISS:
                break
        g += step
    return out


def build_tracklets(tracks: np.ndarray, cuts: list[int], n: int, dets: Dets):
    """Returns dict key -> {"frames": (k,), "boxes": (k,4), "n_bot": int, "n_ext": int, "extra": bool}."""
    cut_arr = np.asarray(sorted(set([0] + [c for c in cuts if 0 < c < n]) | {n}), dtype=np.int64)
    shot_of = lambda f: int(np.searchsorted(cut_arr, f, side="right")) - 1  # noqa: E731
    sub = _subtracks(tracks, cuts)
    trk_by_frame: dict = {}
    for f, tid, x1, y1, x2, y2, _ in tracks:
        trk_by_frame.setdefault(int(f), []).append((int(tid), np.array([x1, y1, x2, y2], np.float64)))
    out = {}
    for key, d in sub.items():
        fr, bx = d["frames"], d["boxes"]
        s = shot_of(int(fr[0]))
        lo, hi = int(cut_arr[s]), int(cut_arr[s + 1]) - 1
        back = _chain(bx[0], int(fr[0]), lo, -1, dets, trk_by_frame, key[0])
        fwd = _chain(bx[-1], int(fr[-1]), hi, +1, dets, trk_by_frame, key[0])
        frames = [g for g, _ in back[::-1]] + list(map(int, fr)) + [g for g, _ in fwd]
        boxes = [b for _, b in back[::-1]] + list(bx) + [b for _, b in fwd]
        core = np.r_[np.zeros(len(back), bool), np.ones(len(fr), bool), np.zeros(len(fwd), bool)]
        out[key] = {"frames": np.asarray(frames), "boxes": np.asarray(boxes, np.float64), "core": core, "n_bot": int(len(fr)),
                    "n_ext": len(back) + len(fwd), "n_back": len(back), "extra": False, "shot": s}
    # extra tracklets from detections no tracklet covers
    covered: dict = {}
    for key, d in out.items():
        for f, b in zip(d["frames"], d["boxes"]):
            covered.setdefault(int(f), []).append(b)
    nxt = 1_000_000
    for s in range(len(cut_arr) - 1):
        active = []    # [frames, boxes, misses]
        done = []
        for f in range(int(cut_arr[s]), int(cut_arr[s + 1])):
            c, cf = dets.at(f)
            if len(c):
                keep = cf >= EXTRA_MIN_CONF
                if f in covered and keep.any():
                    cov = np.asarray(covered[f])
                    keep &= np.array([_iou(cov, b).max() < COVER_IOU for b in c])
                c = c[keep]
            used = np.zeros(len(c), bool)
            for a in sorted(active, key=lambda a: -len(a[0])):
                if len(c) == 0:
                    a[2] += 1
                    continue
                iou = _iou(c, a[1][-1])
                iou[used] = 0
                j = int(np.argmax(iou))
                hr = (c[j, 3] - c[j, 1]) / max(a[1][-1][3] - a[1][-1][1], 1.0)
                if iou[j] >= EXT_IOU and 1 / EXT_HRATIO <= hr <= EXT_HRATIO:
                    a[0].append(f)
                    a[1].append(c[j].copy())
                    a[2] = 0
                    used[j] = True
                else:
                    a[2] += 1
            done += [a for a in active if a[2] > EXT_MAX_MISS]
            active = [a for a in active if a[2] <= EXT_MAX_MISS]
            for j in np.where(~used)[0]:
                active.append([[f], [c[j].copy()], 0])
        done += active
        for a in done:
            if len(a[0]) >= LOCK_MIN_FRAMES:
                out[(nxt, s)] = {"frames": np.asarray(a[0]), "boxes": np.asarray(a[1], np.float64),
                                 "core": np.zeros(len(a[0]), bool), "n_bot": 0,
                                 "n_ext": len(a[0]), "n_back": 0, "extra": True, "shot": s}
                nxt += 1
    return out


def lock_offline(video: Path, tracks: np.ndarray, cuts: list[int], n: int, fps: float, reid_model, H: int, W: int, dets: Dets):
    trk = build_tracklets(tracks, cuts, n, dets)
    min_frames = int(round(LOCK_MIN_SEC * fps))
    half_diag = 0.5 * np.hypot(W, H)
    seed, best = None, 0.0
    for key, d in trk.items():          # seed: same rule as the online lock (on the BoT-SORT part only)
        if d["extra"] or d["n_bot"] < min_frames:
            continue
        b = d["boxes"]
        if not _looks_like_skater(b, H, LOCK_SEED_MIN_H):
            continue
        area = np.maximum(0.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / (W * H)
        dist = np.hypot((b[:, 0] + b[:, 2]) / 2 - W / 2, (b[:, 1] + b[:, 3]) / 2 - H / 2)
        sc = float((area * np.maximum(0.0, 1.0 - dist / half_diag)).sum())
        if sc > best:
            seed, best = key, sc
    info = {"method": "lock_offline", "seed": None, "attached": [], "rejected_top": [], "n_tracklets": len(trk),
            "n_extra_tracklets": int(sum(d["extra"] for d in trk.values()))}
    box: list = [None] * n
    ids: list = [None] * n
    if seed is None:
        return box, ids, info

    def sample(key, k):
        d = trk[key]
        idx = np.unique(np.linspace(0, len(d["frames"]) - 1, k).round().astype(int))
        return [(int(d["frames"][i]), d["boxes"][i].astype(np.float32)) for i in idx]

    cands = [k for k in trk if k != seed and len(trk[k]["frames"]) >= LOCK_MIN_FRAMES
             and _looks_like_skater(trk[k]["boxes"], H, LOCK_ATTACH_MIN_H)]
    wanted: dict = {}
    for f, b in sample(seed, SEED_SAMPLES):
        wanted.setdefault(f, []).append((("seed",), b))
    for k in cands:
        for f, b in sample(k, TRK_SAMPLES):
            wanted.setdefault(f, []).append((k, b))
    emb = _embed_samples(video, wanted, reid_model)
    gal = emb[("seed",)]
    s1 = {k: float((emb[k] @ gal.T).mean()) for k in cands if k in emb}
    add = [k for k in s1 if s1[k] >= GALLERY_ADD_SIM]
    gal2 = np.concatenate([gal] + [emb[k] for k in add]) if add else gal
    if len(gal2) > GALLERY_MAX:
        gal2 = gal2[np.unique(np.linspace(0, len(gal2) - 1, GALLERY_MAX).round().astype(int))]
    s2 = {k: float((emb[k] @ gal2.T).mean()) for k in s1}
    score = {k: max(s1[k], s2[k]) for k in s1}
    info["gallery_size"] = [int(len(gal)), int(len(gal2))]

    def put(key, part):
        """part 'core': the BoT-SORT frames; 'ext': the frames added by extension (only into still-empty frames)."""
        d = trk[key]
        for f, b, c in zip(d["frames"], d["boxes"], d["core"]):
            if c == (part == "core") and 0 <= f < n and box[f] is None:
                box[f], ids[f] = [float(v) for v in b], int(key[0])

    put(seed, "core")
    d = trk[seed]
    info["seed"] = {"id": seed[0], "shot": int(d["shot"]), "frames": [int(d["frames"][0]), int(d["frames"][-1])],
                    "n": int(len(d["frames"])), "n_ext": d["n_ext"], "score": round(best, 2)}
    accepted = []
    for k in sorted(score, key=lambda k: -score[k]):
        d = trk[k]
        bx = d["boxes"]
        dominant = (float(np.median(bx[:, 3] - bx[:, 1])) / H >= 0.30
                    and abs(float(np.median((bx[:, 0] + bx[:, 2]) / 2)) - W / 2) <= 0.3 * W)
        rec = {"id": int(k[0]), "shot": int(d["shot"]), "frames": [int(d["frames"][0]), int(d["frames"][-1])],
               "n": int(len(d["frames"])), "n_ext": d["n_ext"], "n_back": d["n_back"], "extra": d["extra"],
               "sim": round(score[k], 3), "sim_seed": round(s1[k], 3), "sim_gallery": round(s2[k], 3), "dominant": bool(dominant)}
        if score[k] >= LOCK_REID_SIM or (dominant and score[k] >= LOCK_REID_SIM_DOMINANT):
            accepted.append(k)
            info["attached"].append(rec)
        elif len(info["rejected_top"]) < 10:
            info["rejected_top"].append(rec)
    # Fill order (a frame keeps the first box it gets): BoT-SORT parts of the seed and of accepted tracks (most similar first),
    # then their extensions, then extra tracklets built from leftover detections. Extensions and extra tracklets therefore
    # only fill frames nobody else covers (456: an extension and an extra tracklet that ran onto spectators had overridden
    # the skater when everything was filled by similarity alone).
    for k in accepted:
        if not trk[k]["extra"]:
            put(k, "core")
    put(seed, "ext")
    for k in accepted:
        if not trk[k]["extra"]:
            put(k, "ext")
    for k in accepted:
        if trk[k]["extra"]:
            put(k, "ext")
    return box, ids, info


def run(video: Path, track_dir: Path, out_dir: Path, device="cuda", margin=1.12, detector="medium") -> dict:
    from reid import ReIDModel

    old = json.loads((track_dir / "boxes.json").read_text())
    W, H, n, fps = int(old["width"]), int(old["height"]), int(old["n_frames"]), float(old["fps"])
    cuts_pairs = old.get("shot_cuts", [])
    cuts = [int(c) for c, _ in cuts_pairs]
    tracks = np.loadtxt(track_dir / "tracks.txt", delimiter=",").reshape(-1, 7)
    dets = Dets(np.load(track_dir / f"detections_{detector}.npz"), n)
    t0 = time.perf_counter()
    reid_model = ReIDModel.from_pretrained("osnet_x1_0_sportsmot", device=device)
    raw, ids, info = lock_offline(video, tracks, cuts, n, fps, reid_model, H, W, dets)
    t_lock = time.perf_counter() - t0
    present = apply_presence(raw, cuts, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"width": W, "height": H, "n_frames": n, "fps": fps, "margin": margin, "detector": detector,
               "tracker": old.get("tracker"), "target": "lock_offline", "present": [bool(v) for v in present],
               "shot_cuts": cuts_pairs, "lock": info, "lock_seconds": round(t_lock, 2),
               "track_seconds": old.get("track_seconds"), "det_seconds": old.get("det_seconds"),
               "source_track_dir": str(track_dir)}
    if not present.any():
        payload["no_person"] = True
        (out_dir / "boxes.json").write_text(json.dumps(payload))
        return payload
    tight, collapsed, gaps = B.repair_boxes(raw, W, H)
    key = f"padded_{int(round(margin * 100))}"
    payload.update({"collapsed_repaired_frames": collapsed, "gap_interpolated_frames": gaps, "raw_D": raw, "raw_D_ids": ids,
                    "tight_clean": [[round(v, 2) for v in b] for b in tight], key: B.expand_boxes(tight, margin, W, H),
                    "padded_112": B.expand_boxes(tight, 1.12, W, H), "padded_125": B.expand_boxes(tight, 1.25, W, H)})
    (out_dir / "boxes.json").write_text(json.dumps(payload))
    with (out_dir / "boxes_padded_112.txt").open("w") as f:
        for b in payload["padded_112"]:
            f.write(f"{b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n")
    return payload


def main():
    ap = argparse.ArgumentParser(description="offline per-shot lock-on of one skater from cached detections + tracks")
    ap.add_argument("--video", required=True)
    ap.add_argument("--track-dir", required=True, help="dir with boxes.json (shot cuts), tracks.txt, detections_medium.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    p = run(Path(a.video), Path(a.track_dir), Path(a.out_dir), a.device)
    n = p["n_frames"]
    print(json.dumps({"present_ratio": round(sum(p["present"]) / n, 4), "lock_sec": p["lock_seconds"],
                      "attached": len(p["lock"]["attached"]), "tracklets": p["lock"].get("n_tracklets")}))


if __name__ == "__main__":
    main()
