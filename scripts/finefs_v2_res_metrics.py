"""Experiment 3 of the FineFS v2 test: crops from 0.5x frames (half) vs from full-resolution frames (full).
No 3D ground truth exists, so only indirect measures are reported (none of them is an 'error'):
  conf        mean ViTPose confidence over the 17 points; low_conf = share of points with confidence < 0.3
  reproj      SMPL joints (camera space) projected with K vs the variant's own ViTPose points (12 limb points with
              confidence >= 0.5), median over frames, in % of the box height. SMPL and COCO joints are defined
              differently (e.g. hips), so there is a constant offset; only the half/full difference is meaningful.
  reproj_x    the same projection compared against the OTHER variant's 2D (cross-check that a lower 'reproj' is not
              only the 2D and 3D agreeing with each other)
  jit3d       mean |acceleration| of the 24 SMPL joints in camera space, m/s^2 (inside segments)
  jit2d       mean |acceleration| of the 17 ViTPose points, % box height per frame^2 (points with conf >= 0.5)
  bone_cv     coefficient of variation of 8 limb lengths over each segment, %, frame-weighted mean
  speed       seconds of ViTPose + HMR2 per variant, extra seconds for full-resolution decoding/cropping
  diff        mean distance between the two variants' camera-space joints after removing the pelvis, cm

  python scripts/finefs_v2_res_metrics.py --root outputs/finefs_v2_test --ids 36,261
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SMPL_FROM_COCO = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}  # coco -> smpl24
BONES = [(1, 4), (2, 5), (4, 7), (5, 8), (16, 18), (17, 19), (18, 20), (19, 21)]


def load(p):
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def seg_list(d):
    return [(int(a), int(b)) for a, b, _ in d["segments"]]


def project(j, K):
    u = j @ K.T
    return u[..., :2] / np.maximum(u[..., 2:3], 1e-6)


def metrics(g, kp, kp_other, boxh, fps):
    co, sm = list(SMPL_FROM_COCO), list(SMPL_FROM_COCO.values())
    pres = g["present"]
    conf = kp[pres, :, 2]
    uv = project(g["joints24_cam"][:, sm], g["K"])
    out = {"conf": float(np.nanmean(conf)), "low_conf": float(np.nanmean(conf < 0.3))}
    for name, ref in (("reproj", kp), ("reproj_x", kp_other)):
        e = np.linalg.norm(uv - ref[:, co, :2], axis=-1)
        w = ref[:, co, 2] >= 0.5
        e = np.where(w, e, np.nan)
        per = np.nanmean(e, axis=1) / boxh * 100
        out[name] = float(np.nanmedian(per[pres]))
    a3, a2, cv, wsum = [], [], 0.0, 0
    for a, b in seg_list(g):
        j = g["joints24_cam"][a:b]
        if b - a >= 3:
            a3.append(np.linalg.norm(j[2:] - 2 * j[1:-1] + j[:-2], axis=-1).mean(1) * fps * fps)
            k = kp[a:b]
            acc = np.linalg.norm(k[2:, :, :2] - 2 * k[1:-1, :, :2] + k[:-2, :, :2], axis=-1)
            ok = (k[2:, :, 2] >= 0.5) & (k[1:-1, :, 2] >= 0.5) & (k[:-2, :, 2] >= 0.5)
            a2.append(np.nanmean(np.where(ok, acc, np.nan), axis=1) / boxh[a + 1:b - 1] * 100)
        L = np.stack([np.linalg.norm(j[:, p] - j[:, q], axis=-1) for p, q in BONES], 1)
        cv += float((L.std(0) / L.mean(0)).mean() * 100) * (b - a)
        wsum += b - a
    out["jit3d"] = float(np.mean(np.concatenate(a3))) if a3 else float("nan")
    out["jit2d"] = float(np.nanmean(np.concatenate(a2))) if a2 else float("nan")
    out["bone_cv"] = cv / max(wsum, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/finefs_v2_test")
    ap.add_argument("--ids", default="36,261,819,739,980,652,187,513,521,456")
    a = ap.parse_args()
    R = Path(a.root)
    rows = {}
    for vid in a.ids.split(","):
        gh, gf = load(R / "3_gvhmr" / vid / "gvhmr_fast.npz"), load(R / "_exp3_resolution" / vid / "gvhmr_full.npz")
        kh, kf = load(R / "2_pose2d" / vid / "vitpose_h.npz")["kp2d"], load(R / "_exp3_resolution" / vid / "pose2d_full.npz")["kp2d"]
        bj = json.loads((R / "1_tracking" / vid / "boxes.json").read_text())
        n = len(gh["present"])
        tb = np.asarray(bj["tight_clean"], np.float64)[:n]
        boxh = np.maximum(tb[:, 3] - tb[:, 1], 1.0)
        fps = float(gh["fps"])
        st = json.loads((R / "_exp3_resolution" / vid / "stats.json").read_text())
        t = st["timing_sec"]
        r = {"res": f"{int(gh['width'])}x{int(gh['height'])}", "box_h_px_median": float(np.median(boxh[gh["present"]])),
             "half": metrics(gh, kh, kf, boxh, fps), "full": metrics(gf, kf, kh, boxh, fps)}
        r["half"]["net_sec"] = t["half_vitpose"] + t["half_hmr2"]
        r["full"]["net_sec"] = t["full_vitpose"] + t["full_hmr2"]
        r["full_crop_extra_sec"] = t["crop_full"]
        r["video_sec"] = st["video_sec"]
        jh, jf = gh["joints24_cam"], gf["joints24_cam"]
        p = gh["present"]
        d = np.linalg.norm((jh[p] - jh[p, :1]) - (jf[p] - jf[p, :1]), axis=-1).mean() * 100
        r["diff_pelvis_aligned_cm"] = float(d)
        r["diff_abs_cm"] = float(np.linalg.norm(jh[p] - jf[p], axis=-1).mean() * 100)
        rows[vid] = r
        h, f = r["half"], r["full"]
        print(f"{vid} {r['res']} boxh {r['box_h_px_median']:.0f}px | conf {h['conf']:.3f}/{f['conf']:.3f} "
              f"reproj {h['reproj']:.2f}/{f['reproj']:.2f} x {h['reproj_x']:.2f}/{f['reproj_x']:.2f} "
              f"jit3d {h['jit3d']:.1f}/{f['jit3d']:.1f} jit2d {h['jit2d']:.2f}/{f['jit2d']:.2f} "
              f"bone {h['bone_cv']:.2f}/{f['bone_cv']:.2f} net {h['net_sec']:.0f}/{f['net_sec']:.0f}s +crop {r['full_crop_extra_sec']:.0f}s "
              f"diff {r['diff_pelvis_aligned_cm']:.2f}cm")
    (R / "_exp3_resolution" / "metrics.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
