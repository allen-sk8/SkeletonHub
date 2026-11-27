"""Automatic in-field enrollment with no labels ("開場站在鏡頭前 -> 滑進場"), plus causal (live-like) trip scoring.

New file; builds on field_enroll.py (banks, sample score, label-free thresholds). Nothing else changed.

Setting: fixed camera; at the start of a video the user stands in front of the camera (the biggest person), then skates
away into the field. Steps, all from track_person.py tracks.txt (botsort-reid) and enroll_identity.py samples (every 3rd frame):

1. START  = among tracks seen in the first START_WINDOW_S seconds for at least START_MIN_S, the one with the largest median
            box height inside that window. ("ratio" = its height / the runner-up's, reported so a close call is visible.)
2. CHAIN  = follow that track; when it ends, attach the next track only if exactly one track starts within CHAIN_MAX_GAP_S
            whose first box centre is within CHAIN_MAX_DIST_H x (last box height) of the last box centre.
            Stop (end of enrollment) at the first of: no / ambiguous successor; the chain box overlapping another box with
            IoU > CHAIN_STOP_IOU (a tracker may swap identities there; off with stop_on_overlap=False, which trusts the
            tracker through overlaps); ENROLL_MAX_S after the start.
3. USER BANK = usable samples (conf >= 0.5, overlapping no other box, not at the frame edge) of the chain up to the stop.
   OTHERS BANK = tracks seen at the same time as the chain (two boxes on one frame cannot both be the user), only their
   samples up to the stop. Both fixed, never grow.
4. Live scoring of a later trip (one track): its samples in time order; per sample the field_enroll score; the trip score at
   time t = median over up to 15 samples spread over those seen until t (only the past). The trip is judged "user" at the
   first t with >= MIN_SAMPLES samples and score >= threshold, and its end-of-trip score is also kept.

Library: from preprocessors.auto_enroll import pick_start, follow_chain, build_session, live_trip
"""
from __future__ import annotations

import numpy as np

from preprocessors import field_enroll as fe

START_WINDOW_S, START_MIN_S = 5.0, 0.5
CHAIN_MAX_GAP_S, CHAIN_MAX_DIST_H, CHAIN_STOP_IOU = 1.0, 1.5, 0.3
ENROLL_MAX_S = 30.0


def _iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def pick_start(tracks: np.ndarray, fps: float, t0_frame: int = 0) -> dict:
    """tracks.txt rows (frame, id, x1, y1, x2, y2, conf). Returns {tid, height, ratio, candidates}."""
    lo, hi = t0_frame, t0_frame + int(START_WINDOW_S * fps)
    w = tracks[(tracks[:, 0] >= lo) & (tracks[:, 0] < hi)]
    cands = []
    for t in np.unique(w[:, 1]).astype(int):
        r = w[w[:, 1] == t]
        if len(r) >= START_MIN_S * fps:
            cands.append((float(np.median(r[:, 5] - r[:, 3])), t))
    cands.sort(reverse=True)
    if not cands:
        return dict(tid=None, height=None, ratio=None, candidates=[])
    ratio = cands[0][0] / cands[1][0] if len(cands) > 1 else float("inf")
    return dict(tid=int(cands[0][1]), height=cands[0][0], ratio=float(ratio),
                candidates=[dict(tid=int(t), height=h) for h, t in cands[:5]])


def follow_chain(tracks: np.ndarray, start_tid: int, fps: float, stop_on_overlap: bool = True) -> dict:
    """stop_on_overlap=False: keep following the tracker through overlaps (those samples are still never used).
    Returns {segments: [(tid, first_frame, last_frame)], end_frame, stop_reason}."""
    by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(tracks[:, 0]).astype(int):
        by_frame[f] = tracks[tracks[:, 0] == f]
    first = {int(t): tracks[tracks[:, 1] == t][0] for t in np.unique(tracks[:, 1])}
    segs, tid, used = [], start_tid, {start_tid}
    f_start = int(tracks[tracks[:, 1] == start_tid, 0].min())
    f_max = f_start + int(ENROLL_MAX_S * fps)
    while True:
        r = tracks[tracks[:, 1] == tid]
        r = r[r[:, 0] <= f_max]
        s0, last = int(r[0, 0]), None
        reason = None
        for row in r:
            f = int(row[0])
            fr = by_frame[f]
            oth = fr[fr[:, 1] != tid]
            if stop_on_overlap and len(oth) and _iou(row[2:6], oth[:, 2:6]).max() > CHAIN_STOP_IOU:
                reason = "overlap"
                break
            last = row
        if last is None:          # overlapping on its very first frame
            segs.append((tid, s0, s0))
            return dict(segments=segs, end_frame=s0, stop_reason=reason)
        segs.append((tid, s0, int(last[0])))
        if reason:
            return dict(segments=segs, end_frame=int(last[0]), stop_reason=reason)
        if int(last[0]) >= f_max:
            return dict(segments=segs, end_frame=int(last[0]), stop_reason="max_time")
        lf, lb = int(last[0]), last[2:6]
        h = lb[3] - lb[1]
        c = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2])
        succ = []
        for t, fr in first.items():
            if t in used or not (lf < fr[0] <= lf + CHAIN_MAX_GAP_S * fps):
                continue
            cc = np.array([(fr[2] + fr[4]) / 2, (fr[3] + fr[5]) / 2])
            if np.linalg.norm(cc - c) <= CHAIN_MAX_DIST_H * h:
                succ.append(t)
        if len(succ) != 1:
            return dict(segments=segs, end_frame=lf, stop_reason="track_end" if not succ else "ambiguous")
        tid = succ[0]
        used.add(tid)


def chain_mask(d: dict, segments) -> np.ndarray:
    """Sample rows (enroll_feat.npz) that belong to the chain."""
    m = np.zeros(len(d["frame"]), bool)
    for t, a, b in segments:
        m |= (d["tid"] == t) & (d["frame"] >= a) & (d["frame"] <= b)
    return m


def build_session(emb, d: dict, usable: np.ndarray, chain: dict, tracks: np.ndarray, fps: float = 30.0):
    """field_enroll.Session with USER = chain samples, OTHERS = co-present tracks (samples up to the chain end)."""
    cm = chain_mask(d, chain["segments"])
    s = fe.Session.__new__(fe.Session)
    s.user = emb[usable & cm].astype(np.float32)
    s.user_frames = d["frame"][usable & cm]
    s.fps = fps
    chain_tids = {t for t, _, _ in chain["segments"]}
    co = set()
    for t, a, b in chain["segments"]:
        fr = tracks[(tracks[:, 1] == t) & (tracks[:, 0] >= a) & (tracks[:, 0] <= b), 0]
        co |= set(np.unique(tracks[np.isin(tracks[:, 0], fr), 1]).astype(int).tolist())
    co -= chain_tids
    s.others = {}
    for t in sorted(co):
        m = usable & (d["tid"] == t) & (d["frame"] <= chain["end_frame"])
        if m.sum() >= fe.MIN_SAMPLES:
            s.others[f"self:{t}"] = emb[m].astype(np.float32)
    return s, sorted(co)


def _topk_mean(sim: np.ndarray) -> np.ndarray:
    k = min(fe.TOPK, sim.shape[1])
    return np.partition(sim, sim.shape[1] - k, axis=1)[:, -k:].mean(1)


def sample_scores(sess, E: np.ndarray, skip_key: str | None = None, use_others: bool = True) -> np.ndarray:
    """Vectorised field_enroll.Session.sample for many samples: sU - sO (or sU when use_others is False)."""
    sU = _topk_mean(E @ sess.user.T)
    if not use_others:
        return sU
    best = [_topk_mean(E @ o.T) for k, o in sess.others.items() if k != skip_key]
    sO = np.maximum(np.max(best, axis=0), 0.0) if best else np.zeros(len(E), np.float32)
    return sU - sO


def live_trip(sess, emb, d: dict, usable: np.ndarray, tid: int, thr: float, use_others: bool = True,
              skip_key: str | None = None, after_frame: int = -1) -> dict | None:
    """Causal score of one trip. Returns None if < MIN_SAMPLES usable samples, else
    {n, final, first_yes_frame (None if never), series: [(frame, running score)]}."""
    idx = np.where(usable & (d["tid"] == tid) & (d["frame"] > after_frame))[0]
    idx = idx[np.argsort(d["frame"][idx], kind="stable")]
    if len(idx) < fe.MIN_SAMPLES:
        return None
    per = sample_scores(sess, emb[idx].astype(np.float32), skip_key, use_others)
    series, first_yes = [], None
    for k in range(fe.MIN_SAMPLES, len(idx) + 1):
        sc = float(np.median(per[fe.spread(np.arange(k))]))
        f = int(d["frame"][idx[k - 1]])
        series.append((f, sc))
        if first_yes is None and sc >= thr:
            first_yes = f
    return dict(n=int(len(idx)), final=series[-1][1], first_yes_frame=first_yes, series=series,
                start_frame=int(d["frame"][idx[0]]))


# ---------------------------------------------------------------- round-3 improvements (2026-11-28)
CLOSE_H_FRAC, CLOSE_WINDOW_S, CLOSE_MIN_S = 0.40, 30.0, 0.3


def pick_start_close(tracks: np.ndarray, fps: float, frame_h: int) -> dict:
    """START, "close to the camera" version: the FIRST track whose box is at least CLOSE_H_FRAC of the frame height for
    CLOSE_MIN_S (not necessarily consecutive) within the first CLOSE_WINDOW_S seconds. The user walks up to the camera
    at the start; others rarely come that close. Falls back to pick_start when nobody does."""
    big = tracks[(tracks[:, 0] < CLOSE_WINDOW_S * fps) & ((tracks[:, 5] - tracks[:, 3]) >= CLOSE_H_FRAC * frame_h)]
    first = []
    for t in np.unique(big[:, 1]).astype(int):
        r = big[big[:, 1] == t]
        if len(r) >= CLOSE_MIN_S * fps:
            first.append((float(r[int(CLOSE_MIN_S * fps) - 1, 0]), t, float(np.median(r[:, 5] - r[:, 3]))))
    if not first:
        out = pick_start(tracks, fps)
        out["rule"] = "fallback_largest_first5s"
        return out
    first.sort()
    f, t, h = first[0]
    return dict(tid=int(t), height=h, ratio=None, at_frame=int(f), rule="close",
                candidates=[dict(tid=int(x[1]), at_frame=int(x[0]), height=x[2]) for x in first[:5]])


def live_lock(trips: dict, thr: float, end_of: dict, init: tuple | None = None) -> list:
    """"Only one user at a time, and once found, follow him": replay all trips' running scores in time order.
    trips = {tid: [(frame, running score), ...]}, end_of = {tid: last frame of that track}.
    No lock: at each frame, among trips whose latest score >= thr, lock the highest. Locked: keep it until its track ends
    (the tracker follows him; appearance is not re-judged). init = (tid, frame): already locked at the start (e.g. the
    last track of the enrollment chain). Returns lock intervals [(tid, frame_on, frame_off)]."""
    events = sorted((f, t, s) for t, ser in trips.items() for f, s in ser)
    latest: dict = {}
    locks, cur = [], (tuple(init) if init else None)
    i = 0
    while i < len(events):
        f = events[i][0]
        while i < len(events) and events[i][0] == f:
            _, t, s = events[i]
            latest[t] = s
            i += 1
        if cur is not None and f > end_of[cur[0]]:
            locks.append((cur[0], cur[1], end_of[cur[0]]))
            cur = None
        if cur is None:
            cand = [(s, t) for t, s in latest.items() if s >= thr and end_of[t] >= f]
            if cand:
                cur = (max(cand)[1], f)
    if cur is not None:
        locks.append((cur[0], cur[1], end_of[cur[0]]))
    return locks
