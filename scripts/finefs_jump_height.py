"""Jump height for FineFS v2 GVHMR world skeletons (prototype, CPU only).

GVHMR's world trajectory shows jumps only as a small bump (15-50 cm for triples; real ~50-65 cm) and can drift upward
after a jump (the body floats above the ice afterwards). This module:
  1. finds each jump's flight from the body rotation: inside the annotated element time, the run of frames whose yaw
     rate is above half of the peak rate (gaps of <= 2 frames bridged) = flight time T. If that run's turns are fewer
     than 0.6 x the jump's revolutions, the detection is unreliable and a typical flight time for the revolution count
     is used instead (FLIGHT_TYPICAL); its timing is still centred on the peak rotation rate.
  2. re-grounds every frame outside flights: the lowest foot point (ankles and feet) is put on the ice, using a 1 s
     running median so knee bends are kept and drift is removed.
  3. replaces the pelvis height during each flight with: straight line between take-off and landing heights plus a
     parabola of height h = g T^2 / 8 (free fall: T/2 up, T/2 down). The whole body moves with the pelvis (pose kept).
  4. lifts any frame whose lowest foot point would be below the ice.
Only the vertical axis changes. Combination jumps (3Lz+3T) get one flight per part, found peak by peak.

  python scripts/finefs_jump_height.py --id 0          # prints the jumps found and their heights
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.spatial.transform import Rotation as Rot

D = Path("/home/allen/datasets/FineFS")
G = 9.81
FOOT = [7, 8, 10, 11]                         # SMPL ankles and feet
FLIGHT_TYPICAL = {0.5: 0.35, 1: 0.45, 1.5: 0.50, 2: 0.52, 2.5: 0.56, 3: 0.64, 3.5: 0.66, 4: 0.72}   # seconds


def t2s(x):
    m, s = x.split("-")
    return int(m) * 60 + int(s)


def parts_revs(element: str):
    """'3Lz+3T' -> [3, 3]; '3A' -> [3.5]; parts without a number (COMBO, Eu = 1) handled."""
    out = []
    for p in element.split("+"):
        m = re.match(r"(\d)([A-Za-z]+)", p)
        if m:
            n = int(m.group(1))
            out.append(n + 0.5 if m.group(2).upper().startswith("A") else float(n))
        elif p.upper().startswith("EU"):
            out.append(1.0)
    return out


def yaw_rate(global_orient, fps):
    fwd = Rot.from_rotvec(global_orient).as_matrix()[:, :, 2]
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    return yaw, uniform_filter1d(np.abs(np.gradient(yaw)) * fps, 3)


def flight_around(rs, yaw, pk, revs, fps):
    th = 0.5 * rs[pk]
    i = j = pk
    while True:
        while i > 0 and rs[i - 1] > th:
            i -= 1
        if i > 2 and rs[i - 3:i].max() > th:
            i -= 3; continue
        break
    while True:
        while j < len(rs) - 1 and rs[j + 1] > th:
            j += 1
        if j < len(rs) - 3 and rs[j + 1:j + 4].max() > th:
            j += 3; continue
        break
    turns = abs(yaw[j] - yaw[i]) / (2 * np.pi)
    if turns >= 0.6 * revs and (j - i + 1) / fps < 1.0:
        return i, j + 1, "rotation"
    T = FLIGHT_TYPICAL.get(revs, 0.6)
    c = (i + j) / 2
    n = int(round(T * fps))
    return int(round(c - n / 2)), int(round(c - n / 2)) + n, "typical"


def find_flights(g, ann, fps):
    """List of (start, end_exclusive, revs, how, element) in video frames, only inside one GVHMR segment each."""
    seg = g["segment_id"]
    yaw_all, rs_all = None, None
    flights = []
    for e in ann.values():
        if e.get("coarse_class") != "jump":
            continue
        revs = parts_revs(e["element"])
        if not revs:
            continue
        s, t = [t2s(x) for x in e["time"].split(",")]
        f0, f1 = max(0, int((s - 0.5) * fps)), min(len(seg), int((t + 1.5) * fps))
        sg = seg[f0:f1]
        if (sg < 0).mean() > 0.2:
            continue
        main = np.bincount(sg[sg >= 0]).argmax()
        idx = np.where(sg == main)[0]
        f0, f1 = f0 + idx[0], f0 + idx[-1] + 1
        yaw, rs = yaw_rate(g["smpl_world_global_orient"][f0:f1], fps)
        lo = 0
        for r in revs:                                    # one peak per part, left to right
            if lo >= len(rs) - 3:
                break
            pk = lo + int(rs[lo:].argmax()) if r == revs[-1] else lo + int(rs[lo:lo + int(2.5 * fps)].argmax())
            a, b, how = flight_around(rs, yaw, pk, r, fps)
            a, b = max(a, 0), min(b, len(rs))
            flights.append((f0 + a, f0 + b, r, how, e["element"]))
            lo = b + int(0.15 * fps)
    return flights


def fix_vertical(g, flights, fps):
    """Vertical offset dy (T,) to add to transl / every joint's Y (world, metres), per segment."""
    J = g["joints24_world"]
    n = len(J)
    seg = g["segment_id"]
    dy = np.zeros(n, np.float32)
    in_air = np.zeros(n, bool)
    for a, b, *_ in flights:
        in_air[a:b] = True
    low = J[:, FOOT, 1].min(1)
    for sid in np.unique(seg[seg >= 0]):
        idx = np.where(seg == sid)[0]
        gi = idx[~in_air[idx]]
        if len(gi) < 3:
            continue
        ground = np.interp(idx, gi, median_filter(low[gi], size=max(3, int(fps)) | 1, mode="nearest"))
        dy[idx] = -ground
    pel = J[:, 0, 1] + dy
    for a, b, r, how, el in flights:
        if b - a < 3 or seg[a] < 0 or seg[a] != seg[b - 1]:
            continue
        T = (b - a) / fps
        h = G * T * T / 8
        s = np.arange(b - a) / (b - a - 1)
        p0, p1 = pel[a - 1] if a > 0 else pel[a], pel[b] if b < n else pel[b - 1]
        target = p0 + (p1 - p0) * s + 4 * h * s * (1 - s)
        dy[a:b] += target - pel[a:b]
    low_fixed = low + dy                                  # no foot below the ice (flight edges, take-off squat)
    dy += np.maximum(0.0, -low_fixed) * (seg >= 0)
    return dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    a = ap.parse_args()
    g = np.load(D / "3_gvhmr" / a.id / "gvhmr_fast.npz")
    fps = float(g["fps"])
    ann = json.loads((D / "annotation" / f"{a.id}.json").read_text())["executed_element"]
    fl = find_flights(g, ann, fps)
    for f0, f1, r, how, el in fl:
        T = (f1 - f0) / fps
        print(f"{el:12s} part {r:>3} turns  frames {f0}-{f1}  flight {T:.2f} s  height {G * T * T / 8 * 100:.0f} cm  ({how})")


if __name__ == "__main__":
    main()
