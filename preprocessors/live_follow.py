"""Live "find the user, then follow him" simulation, frame by frame (sample frames, e.g. every 3rd), causal.
New file (round 3 of the replay-system re-ID tests, 2026-11-28). Uses enroll_identity.py samples (enroll_feat.npz rows: one per
tracked box on each sample frame) and any embedding (CLIP, OSNet, colour, combined); auto_enroll.py for the start and chain.

State machine, evaluated at every sample frame in time order (only the past is used):
  FOLLOW tid   keep following that track (the tracker carries him, appearance is not re-judged). Released when the track ends,
               or (guard against tracker identity swaps) when the median user-similarity of its last GUARD_N usable samples
               falls below the others level.
               guard="near_floor": also released when the followed person is near and his last GUARD_N samples are below
               the floor (he is big enough to judge and does not look like the user).
               far_release_s: also released when the followed person has been far (box < near_h) for this long:
               far = out of range, he is found again when he comes near.
               grow=True: every usable sample (not overlapping anyone) of the followed track is added to the USER bank, so the
               bank learns how he looks in the field (small, blurred). Only tracking continuity decides what is added.
  SEARCH       only NEAR people are judged (box height >= near_h): far away is treated as "not sure = out of range".
               score of a person = median sU over his last RECENT_N usable samples (sU = mean of the 3 most similar bank samples).
               mode "relative": lock the best candidate if score >= floor and (best - second best near candidate) >= margin.
               mode "absolute": lock the best candidate if score - (most similar known other) >= 0 (the earlier field_enroll rule);
               with floor_frac also score >= others + floor_frac x (self - others).
               grow_gate=True: only near samples whose own similarity is >= floor are added (more conservative).
Thresholds come from the enrollment alone (no labels): self = enrollment samples vs each other (>= 1 s apart),
others = people seen together with the enrollment chain vs the bank; floor = midpoint; margin = MARGIN_FRAC x (self - others).

Library: from preprocessors.live_follow import Follower
"""
from __future__ import annotations

import numpy as np

TOPK, RECENT_N, GUARD_N, MARGIN_FRAC, MIN_N = 3, 5, 10, 0.25, 2
BANK_CAP = 3000


def _topk(sim: np.ndarray) -> np.ndarray:
    k = min(TOPK, sim.shape[-1])
    return np.partition(sim, sim.shape[-1] - k, axis=-1)[..., -k:].mean(-1)


class Follower:
    def __init__(self, user_bank: np.ndarray, others: dict, self_level: float, fps: float = 30.0, near_h: float = 150.0,
                 mode: str = "relative", grow: bool = True, guard: str = "others", min_n: int = MIN_N, grow_gate: bool = False,
                 far_release_s: float | None = None, floor_frac: float | None = None):
        self.bank = [np.asarray(user_bank, np.float32)]
        self.n_enroll = len(user_bank)
        self.others = {k: np.asarray(v, np.float32) for k, v in others.items()}
        self.fps, self.near_h, self.mode, self.grow = fps, near_h, mode, grow
        self.guard, self.min_n, self.grow_gate = guard, min_n, grow_gate
        self.far_release_s, self.far_since = far_release_s, None
        B = self.bank[0]
        oth = [float(np.median(_topk(o @ B.T))) for o in self.others.values()] if self.others else []
        self.self_level = self_level
        self.others_level = float(np.median(oth)) if oth else self_level - 0.1
        self.floor = (self.self_level + self.others_level) / 2
        self.margin = MARGIN_FRAC * max(1e-3, self.self_level - self.others_level)
        self.both = floor_frac is not None
        if self.both:   # mode "absolute" + also sU >= others + floor_frac x (self - others)   (0.5 = the midpoint floor)
            self.floor = self.others_level + floor_frac * (self.self_level - self.others_level)
        self.recent: dict[int, list] = {}
        self.cur = None
        self.added = []

    def user(self) -> np.ndarray:
        if len(self.bank) > 1:
            self.bank = [np.concatenate(self.bank)[-BANK_CAP:]]
        return self.bank[0]

    def sU(self, E: np.ndarray) -> np.ndarray:
        return _topk(E @ self.user().T)

    def sO(self, E: np.ndarray) -> np.ndarray:
        if not self.others:
            return np.zeros(len(E), np.float32)
        return np.maximum(np.max([_topk(E @ o.T) for o in self.others.values()], axis=0), 0)

    def step(self, f: int, tids: np.ndarray, h: np.ndarray, usable: np.ndarray, E: np.ndarray) -> dict:
        """One sample frame. tids/h/usable/E: the boxes on this frame. Returns {state, tid, scores {tid: score}}."""
        s = self.sU(E[usable]) if usable.any() else np.zeros(0)
        for t, v in zip(tids[usable], s):
            self.recent.setdefault(int(t), []).append(float(v))
            self.recent[int(t)] = self.recent[int(t)][-max(RECENT_N, GUARD_N):]
        if self.cur is not None:
            if self.cur not in set(tids.tolist()):
                self.cur = None                       # track ended / lost
            else:
                rec = self.recent.get(self.cur, [])[-GUARD_N:]
                if len(rec) >= GUARD_N and np.median(rec) < self.others_level:
                    self.cur = None                   # looks like someone else now: tracker swap guard
                elif self.guard == "near_floor" and len(rec) >= GUARD_N and np.median(rec) < self.floor \
                        and (h[tids == self.cur] >= self.near_h).any():
                    self.cur = None                   # near (big enough to judge) but does not look like him: let go
        if self.cur is not None and self.far_release_s is not None:
            hh = h[tids == self.cur]
            if len(hh) and hh.max() < self.near_h:
                self.far_since = f if self.far_since is None else self.far_since
                if f - self.far_since >= self.far_release_s * self.fps:
                    self.cur, self.far_since = None, None   # far away for a while: treat as out of range
            else:
                self.far_since = None
        if self.cur is not None:
            if self.grow:
                m = usable & (tids == self.cur)
                if self.grow_gate:                    # only near samples that themselves look like him
                    m &= (h >= self.near_h)
                    if m.any():
                        m[m] = self.sU(E[m]) >= self.floor
                if m.any():
                    self.bank.append(E[m].astype(np.float32)); self.added.append((f, int(m.sum())))
            return dict(state="follow", tid=self.cur)
        near = usable & (h >= self.near_h)
        cands = []
        for t in np.unique(tids[near]).astype(int):
            rec = self.recent.get(int(t), [])[-RECENT_N:]
            if len(rec) >= self.min_n:
                cands.append((float(np.median(rec)), int(t)))
        cands.sort(reverse=True)
        if cands:
            best, t = cands[0]
            if self.mode == "relative":
                second = cands[1][0] if len(cands) > 1 else -1.0
                ok = best >= self.floor and best - second >= self.margin
            else:
                e = E[(tids == t) & usable]
                ok = best - float(np.median(self.sO(e))) >= 0 and (not self.both or best >= self.floor)
            if ok:
                self.cur = t
                return dict(state="lock", tid=t, score=best)
        return dict(state="search", tid=None, best=cands[0] if cands else None)


def self_level(bank: np.ndarray, frames: np.ndarray, fps: float = 30.0, min_gap_s: float = 1.0) -> float:
    out = []
    for i in range(len(bank)):
        far = np.abs(frames - frames[i]) >= min_gap_s * fps
        if far.sum() >= TOPK:
            out.append(float(_topk(bank[far] @ bank[i])))
    return float(np.median(out)) if out else float("nan")
