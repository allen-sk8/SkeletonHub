"""Jump height measured directly in the image (FineFS v2, prototype, CPU only). Second method next to
scripts/finefs_jump_height.py (flight time from body rotation + parabola); both are kept for comparison / ablation.

For each jump part found by finefs_jump_height.find_flights (which gives the time of fastest rotation):
  1. background homographies between consecutive frames (finefs_ground_track.background_homographies) are chained to
     a reference frame (the fastest-rotation frame), so every frame's ankle position is expressed in that frame's
     pixels with the camera's pan / tilt / zoom removed ("stabilised" image).
  2. lowest ankle (COCO ankles from 2_pose2d, confidence >= 0.3; the larger image y of the two) in stabilised pixels.
  3. the on-ice track: a straight line fitted (least squares) to the stabilised ankle height in 0.25-0.9 s before and
     after the reference frame (the skater glides on the ice there; the line absorbs moving towards / away).
  4. lift(t) = line(t) - ankle_y(t) in pixels -> metres with GVHMR's pelvis depth z at the reference frame:
     m = px * z / (f * cos(elevation)), elevation = ELEV_DEG of finefs_ground_track (camera looks down).
  5. take-off / landing: where lift crosses LIFT_ON (5 cm) around the maximum, linearly interpolated between frames
     (sub-frame). Flight time T_img = landing - take-off. Height h_img = maximum lift (ankle apex above the ice line).
     Also h_from_T = g T_img^2 / 8 (free fall from the measured flight time) as a cross-check.
A jump is marked unreliable when a homography in the window failed, the ankles are missing for > 20% of the frames,
the flight is shorter than 0.2 s or longer than 1.1 s, or the lift does not come back down within the window.

  python scripts/finefs_jump_image.py --ids 0,8,10           # prints one line per jump part
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finefs_ground_track import ELEV_DEG, background_homographies  # noqa: E402
from finefs_jump_height import D, G, find_flights  # noqa: E402

LIFT_ON = 0.05        # m, foot counted as off the ice above this
HALF_WIN = 1.2        # s, window around the fastest-rotation frame


def measure_jump(g, bj, kp, c, fps, video):
    f = float(g["K"][0, 0])
    n_all = len(g["present"])
    w = int(HALF_WIN * fps)
    f0, f1 = max(0, c - w), min(n_all, c + w + 1)
    boxes = [bj["tight_clean"][t] if bj["present"][t] else None for t in range(len(bj["present"]))]
    Hs, _ = background_homographies(video, boxes, f0, f1)
    n = f1 - f0
    r = c - f0
    out = {"ok": False}
    # chain: M[k] maps frame f0+k pixels -> reference frame pixels
    M = np.full((n, 3, 3), np.nan)
    M[r] = np.eye(3)
    for k in range(r + 1, n):                      # H[k]: k-1 -> k ; so k -> ref = M[k-1] @ inv(H[k])
        if np.isfinite(Hs[k]).all() and np.isfinite(M[k - 1]).all():
            M[k] = M[k - 1] @ np.linalg.inv(Hs[k])
    for k in range(r - 1, -1, -1):                 # k -> ref = M[k+1] @ H[k+1]
        if np.isfinite(Hs[k + 1]).all() and np.isfinite(M[k + 1]).all():
            M[k] = M[k + 1] @ Hs[k + 1]
    if not np.isfinite(M).all():
        out["why"] = "background homography failed in the window"
        return out
    a = kp[f0:f1, [15, 16]]                        # (n,2,3) ankles
    y = np.full(n, np.nan); x = np.full(n, np.nan)
    for k in range(n):
        good = [j for j in range(2) if np.isfinite(a[k, j, 0]) and a[k, j, 2] >= 0.3]
        if good:
            j = max(good, key=lambda j: a[k, j, 1])          # lowest ankle in the image
            p = M[k] @ np.array([a[k, j, 0], a[k, j, 1], 1.0])
            x[k], y[k] = p[:2] / p[2]
    if np.isnan(y).mean() > 0.2:
        out["why"] = "ankles missing"
        return out
    t = np.arange(n)
    ok = np.isfinite(y)
    y = np.interp(t, t[ok], y[ok])
    z = float(np.nanmedian(g["joints24_cam"][max(0, c - 3):c + 4, 0, 2]))
    mpp = z / (f * np.cos(np.radians(ELEV_DEG)))  # metres per reference-frame pixel (vertical lift)
    if METHOD == "contact":
        return _contact_method(out, y, t, n, r, mpp, fps, f0, z)
    lo, hi = int(0.25 * fps), int(0.9 * fps)
    base_idx = np.r_[max(0, r - hi):max(0, r - lo), min(n, r + lo):min(n, r + hi)]
    if len(base_idx) < 6:
        out["why"] = "window too short"
        return out
    A = np.c_[base_idx, np.ones(len(base_idx))]
    coef, *_ = np.linalg.lstsq(A, y[base_idx], rcond=None)
    line = coef[0] * t + coef[1]
    lift = (line - y) * mpp
    near = slice(max(0, r - int(0.5 * fps)), min(n, r + int(0.5 * fps)))
    pk = near.start + int(np.argmax(lift[near]))
    if lift[pk] < LIFT_ON:
        out["why"] = "no lift above 5 cm near the rotation"
        return out
    i = pk
    while i > 0 and lift[i - 1] > LIFT_ON:
        i -= 1
    j = pk
    while j < n - 1 and lift[j + 1] > LIFT_ON:
        j += 1
    if i == 0 or j == n - 1:
        out["why"] = "lift does not come back down in the window"
        return out
    # sub-frame crossings
    t_off = (i - 1) + (LIFT_ON - lift[i - 1]) / (lift[i] - lift[i - 1])
    t_on = j + (lift[j] - LIFT_ON) / (lift[j] - lift[j + 1])
    T = (t_on - t_off) / fps
    out.update(t_takeoff=f0 + t_off, t_landing=f0 + t_on, T_img=T, h_img=float(lift[pk]), h_from_T=G * T * T / 8,
               apex_frame=f0 + pk, z=z, lift=lift, f0=f0)
    if not 0.2 <= T <= 1.1:
        out["why"] = f"flight {T:.2f} s out of range"
        return out
    out["ok"] = True
    return out


METHOD = "line"        # "line" (default): straight line fitted 0.25-0.9 s before and after the apex;
                       # "contact": ice line through take-off and landing found from ankle speed (tested 2026-11-24:
                       # worse - flights 0.1-0.2 s too long, height and flight time agree for 26% of jumps vs 59%)
V_FRAC = 0.15          # take-off / landing = where the ankle's vertical speed falls below this fraction of its peak


def _contact_method(out, y, t, n, r, mpp, fps, f0, z):
    """Take-off: going back from the apex, the last frame where the ankle still rises fast; landing: going forward,
    the first frame where it stops coming down. The ice is the straight line between the ankle at those two moments."""
    from scipy.ndimage import uniform_filter1d
    ys = uniform_filter1d(y, 3)
    near = slice(max(0, r - int(0.5 * fps)), min(n, r + int(0.5 * fps)))
    ap = near.start + int(np.argmin(ys[near]))                  # apex: highest ankle in the stabilised image
    up = -np.gradient(ys)                                       # px/frame, positive = moving up
    vmax_up = up[max(0, ap - int(0.5 * fps)):ap + 1].max()
    vmax_dn = (-up[ap:min(n, ap + int(0.5 * fps))]).max()
    if vmax_up <= 0 or vmax_dn <= 0:
        out["why"] = "no rise / fall around the apex"
        return out
    i = ap
    while i > 0 and not (up[i] > V_FRAC * vmax_up and up[i - 1] <= V_FRAC * vmax_up and i < ap - 1):
        i -= 1
    j = ap
    while j < n - 1 and not (-up[j] > V_FRAC * vmax_dn and -up[j + 1] <= V_FRAC * vmax_dn and j > ap + 1):
        j += 1
    if i <= 0 or j >= n - 1:
        out["why"] = "take-off or landing not found in the window"
        return out
    # sub-frame: interpolate the speed threshold crossing
    th_u, th_d = V_FRAC * vmax_up, V_FRAC * vmax_dn
    t_off = (i - 1) + (th_u - up[i - 1]) / (up[i] - up[i - 1])
    t_on = j + (-up[j] - th_d) / ((-up[j]) - (-up[j + 1]))
    line = ys[i - 1] + (ys[j + 1] - ys[i - 1]) * (t - (i - 1)) / ((j + 1) - (i - 1))
    lift = (line - ys) * mpp
    T = (t_on - t_off) / fps
    out.update(t_takeoff=f0 + t_off, t_landing=f0 + t_on, T_img=T, h_img=float(lift[ap]), h_from_T=G * T * T / 8,
               apex_frame=f0 + ap, z=z, lift=lift, f0=f0, method="contact")
    if not 0.2 <= T <= 1.1:
        out["why"] = f"flight {T:.2f} s out of range"
        return out
    if out["h_img"] < 0.1:
        out["why"] = "apex less than 10 cm above the ice line"
        return out
    out["ok"] = True
    return out


def measure_video(vid):
    g = np.load(D / "3_gvhmr" / vid / "gvhmr_fast.npz")
    bj = json.loads((D / "1_tracking" / vid / "boxes.json").read_text())
    kp = np.load(D / "2_pose2d" / vid / "vitpose_h.npz")["kp2d"]
    ann = json.loads((D / "annotation" / f"{vid}.json").read_text())["executed_element"]
    fps = float(g["fps"])
    res = []
    for a, b, revs, how, el in find_flights(g, ann, fps):
        m = measure_jump(g, bj, kp, (a + b) // 2, fps, D / "video" / f"{vid}.mp4")
        m.pop("lift", None)
        res.append({"id": vid, "element": el, "revs": revs, "rot_frames": [int(a), int(b)], "rot_how": how,
                    "T_rot": (b - a) / fps, **{k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) and not isinstance(v, bool) else v) for k, v in m.items()}})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default=None, help="write all results as JSON lines")
    ap.add_argument("--method", default=METHOD, choices=["contact", "line"])
    a = ap.parse_args()
    globals()["METHOD"] = a.method
    rows = []
    for vid in a.ids.split(","):
        for r in measure_video(vid):
            rows.append(r)
            s = (f"T_img {r['T_img']:.2f}s h_img {r['h_img'] * 100:.0f}cm h_from_T {r['h_from_T'] * 100:.0f}cm" if r["ok"]
                 else f"unreliable: {r.get('why')}")
            print(f"{vid:>4} {r['element']:12s} {r['revs']:>3} | rotation {r['T_rot']:.2f}s ({r['rot_how']}) | image {s}", flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
