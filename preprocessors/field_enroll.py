"""In-field enrollment ("場上登錄"): the user is pointed out once (one track = the enrollment trip); that trip's own samples,
at field size, become a FIXED user bank. Every other trip is scored by appearance against it. New file; nothing else changed.

Compared with enroll_identity.py (close-up enrollment video, banks that grow), this one:
  - takes the user bank from one track of the session itself (only samples not overlapping any other box);
  - builds the OTHERS bank only from trips that are certainly not the user: tracks visible at the same time as the
    enrollment track (two boxes on one frame cannot both be the user). No labels needed. Fixed, never grows;
  - works with any embedding: OSNet (enroll_feat.npz "emb_*"), crop_embed.py vectors (DINOv2, CLIP, ...), or body colours.

Per trip (up to 15 usable samples spread in time; usable = conf >= 0.5, no overlap, not at the frame edge):
    sU = mean of the 3 most similar user-bank samples (cosine);  sO = most similar other person (mean of its top 3), 0 if none
    trip score "diff" = median(sU - sO);  "user_only" = median(sU)
A trip is never compared with itself (a co-present trip being scored is left out of the OTHERS bank).

Thresholds set in advance, without labels:
    diff      >= 0                           ("looks more like the user than like anyone known not to be the user")
    user_only >= midpoint(self, others)      self   = median over enrollment samples of sU against the enrollment samples
                                                      at least 1 s away (same trip, so this is optimistic);
                                             others = median trip sU of the co-present trips.
Library: from preprocessors.field_enroll import Session, score_trips.
"""
from __future__ import annotations

import numpy as np

TOPK, MAX_SAMPLES, MIN_SAMPLES = 3, 15, 3
SAMPLE_MIN_CONF, EDGE_PX = 0.5, 5


def usable_mask(d: dict) -> np.ndarray:
    """d = enroll_feat.npz contents (box, conf, occl, width, height)."""
    b, W, H = d["box"], int(d["width"]), int(d["height"])
    edge = (b[:, 0] > EDGE_PX) & (b[:, 2] < W - EDGE_PX) & (b[:, 1] > EDGE_PX) & (b[:, 3] < H - EDGE_PX)
    return (d["conf"] >= SAMPLE_MIN_CONF) & (~d["occl"].astype(bool)) & edge


def color_vectors(d: dict) -> np.ndarray:
    """Upper + lower body HSV histograms as one unit vector; cosine = mean Bhattacharyya coefficient of the two parts."""
    v = np.concatenate([np.sqrt(np.clip(d["up"], 0, None)), np.sqrt(np.clip(d["lo"], 0, None))], 1)
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def combine(*embs: np.ndarray) -> np.ndarray:
    """Concatenate unit vectors; cosine of the result = mean of the parts' cosines."""
    return (np.concatenate([e.astype(np.float32) for e in embs], 1) / np.sqrt(len(embs))).astype(np.float32)


def co_present(tracks: np.ndarray, tid: int) -> set[int]:
    """Track ids that share at least one frame with track `tid` (tracks.txt rows: frame, id, x1, y1, x2, y2, conf)."""
    fr = set(tracks[tracks[:, 1] == tid, 0].astype(int).tolist())
    m = np.isin(tracks[:, 0].astype(int), list(fr)) & (tracks[:, 1] != tid)
    return set(np.unique(tracks[m, 1]).astype(int).tolist())


def spread(idx: np.ndarray, k: int = MAX_SAMPLES) -> np.ndarray:
    return idx if len(idx) <= k else idx[np.linspace(0, len(idx) - 1, k).round().astype(int)]


class Session:
    """Banks for one enrollment: user = samples of the enrollment trip, others = one entry per co-present trip."""

    def __init__(self, emb: np.ndarray, tid: np.ndarray, frame: np.ndarray, usable: np.ndarray, enroll_tid: int,
                 other_tids: set[int], fps: float = 30.0):
        self.user = emb[usable & (tid == enroll_tid)].astype(np.float32)
        self.user_frames = frame[usable & (tid == enroll_tid)]
        self.others = {f"self:{t}": emb[spread(np.where(usable & (tid == t))[0], 10 ** 9)].astype(np.float32)
                       for t in sorted(other_tids) if (usable & (tid == t)).sum() >= MIN_SAMPLES}
        self.fps = fps

    def add_others(self, prefix: str, emb, tid, usable, tids):
        """Others from another video (cross-video use: the bank of an earlier video)."""
        for t in sorted(tids):
            m = usable & (tid == t)
            if m.sum() >= MIN_SAMPLES:
                self.others[f"{prefix}:{t}"] = emb[m].astype(np.float32)

    def sample(self, e: np.ndarray, skip_key: str | None = None, use_others: bool = True):
        s = self.user @ e
        sU = float(np.sort(s)[-TOPK:].mean())
        sO = 0.0
        if use_others:
            best = [float(np.sort(o @ e)[-TOPK:].mean()) for k, o in self.others.items() if k != skip_key]
            sO = max(max(best), 0.0) if best else 0.0
        return sU, sO

    def self_level(self, min_gap_s: float = 1.0) -> float:
        """Median sU of enrollment samples against enrollment samples >= min_gap_s away (optimistic: same trip)."""
        out = []
        for i in range(len(self.user)):
            far = np.abs(self.user_frames - self.user_frames[i]) >= min_gap_s * self.fps
            if far.sum() >= TOPK:
                out.append(float(np.sort(self.user[far] @ self.user[i])[-TOPK:].mean()))
        return float(np.median(out)) if out else float("nan")


def score_trips(sess: Session, emb, tid, usable, trips, key_prefix: str = "self", use_others: bool = True) -> dict:
    """trips: iterable of track ids. Returns {tid: dict(n, sU, sO, diff) or None if < MIN_SAMPLES usable samples}."""
    res = {}
    for t in trips:
        idx = spread(np.where(usable & (tid == t))[0])
        if len(idx) < MIN_SAMPLES:
            res[int(t)] = None
            continue
        ss = np.array([sess.sample(emb[i].astype(np.float32), f"{key_prefix}:{t}", use_others) for i in idx])
        res[int(t)] = dict(n=int(len(idx)), sU=float(np.median(ss[:, 0])), sO=float(np.median(ss[:, 1])),
                           diff=float(np.median(ss[:, 0] - ss[:, 1])))
    return res


def user_only_threshold(sess: Session, co_scores: dict) -> float:
    """Label-free threshold for the user_only score: midpoint of the enrollment self level and the co-present trips' level."""
    oth = [r["sU"] for r in co_scores.values() if r is not None]
    selfl = sess.self_level()
    return float((selfl + np.median(oth)) / 2) if oth else selfl
