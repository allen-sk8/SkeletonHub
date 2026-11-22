"""Compare gvhmr_fast on a WHOLE match video (sliced to the 5 s clip's time span) with GVHMR run directly on the 5 s clip.

  python scripts/finefs_clip_compare.py --ids 36,261 --out outputs/finefs_full_test --device cuda:0     (skeleton_env, 1 GPU)

Needs <out>/<id>/{boxes.json, gvhmr_fast.npz, clips.json} (clips.json from finefs_clip_align.py).
For every 5 s clip a clip dir <out>/clipcmp/<id>_<clip>/ is made with the FineFS_5s clip.mp4 and the SAME boxes (the locked
whole-video boxes sliced to the clip's frames), then run:
   orig    = hmr_zoo's original GVHMR adapter (fp32, flip test, GVHMR's own preprocessing, SimpleVO on the clip)
   precise = gvhmr_fast code with fp32 + flip test (gives pose parameters, so angle differences can be computed)
and compared with the slice of the whole-video gvhmr_fast result ('full'), camera-space H36M-17 joints, frames where 'full' is present.
Writes <out>/clip_compare.json (+ per-clip rows).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path = [p for p in sys.path if ".local" not in p] + [p for p in sys.path if ".local" in p]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402


def mj(a, b, align=False):
    if align:
        a, b = a - a[:, :1], b - b[:, :1]
    return float(np.linalg.norm(a - b, axis=-1).mean() * 100)


def ang(p, q):
    T = p.shape[0]
    a = np.degrees((R.from_rotvec(p.reshape(-1, 3).astype(np.float64)).inv() * R.from_rotvec(q.reshape(-1, 3).astype(np.float64))).magnitude()).reshape(T, 21)
    return float(a.mean()), float(np.percentile(a, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    os.chdir(str(ROOT / "external" / "gvhmr"))
    import torch
    from hmr_zoo import common as C
    from hmr_zoo.adapters import GVHMRDetector as Orig
    from hmr_zoo.fast_io import load_fast
    from detectors.gvhmr_fast_detector import GVHMRFastDetector, save_fast_npz

    orig = Orig(a.device)
    fast = GVHMRFastDetector(a.device, fp16=False, flip_test=True)   # precise mode
    rows = []
    for vid in [v for v in a.ids.split(",") if v]:
        d = out / vid
        if not (d / "gvhmr_fast.npz").exists():
            continue
        full = load_fast(d / "gvhmr_fast.npz")
        bj = json.loads((d / "boxes.json").read_text())
        for c in json.loads((d / "clips.json").read_text()):
            s, n = c["start_frame"], c["n_frames"]
            sl = slice(s, s + n)
            pres = full["present"][sl]
            row = {"video": vid, "clip": c["clip"], "element": c["element"], "start_frame": s, "n_frames": n,
                   "present_in_full": int(pres.sum())}
            if pres.sum() < 20:
                row["skipped"] = "person absent in most of the clip span (nothing to compare)"
                rows.append(row)
                continue
            cd = out / "clipcmp" / f"{vid}_{c['clip']}"
            cd.mkdir(parents=True, exist_ok=True)
            if not (cd / "clip.mp4").exists():
                shutil.copy2(c["clip_mp4"], cd / "clip.mp4")
            sub = {"width": bj["width"], "height": bj["height"], "fps": bj["fps"], "n_frames": n,
                   "padded_112": bj["padded_112"][sl], "tight_clean": bj["tight_clean"][sl], "present": [True] * n,
                   "note": f"slice of {vid} frames {s}-{s + n} of the whole-video lock boxes; absent frames carry filled boxes"}
            (cd / "boxes.json").write_text(json.dumps(sub))
            (cd / "meta.json").write_text(json.dumps({"sport": "figure_skating", "name": cd.name, "width": bj["width"], "height": bj["height"],
                                                      "fps": bj["fps"], "n_frames": n, "source": {"type": "finefs_full_slice", "video": vid}}))
            r_o = orig.detect_video(str(cd / "clip.mp4"), str(C.Clip(cd).boxes_txt("padded_112")))
            rp = fast.detect_video_fast(cd / "clip.mp4", cd / "boxes.json", verbose=False)
            m = pres & np.isfinite(rp["joints17_cam"][:len(pres), 0, 0])
            T = min(len(m), len(r_o["joints3d_cam"]))
            m = m[:T]
            jf = full["joints17_cam"][sl][:T][m]
            jo = r_o["joints3d_cam"][:T][m]
            jp = rp["joints17_cam"][:T][m]
            row.update({
                "frames_compared": int(m.sum()),
                "full_vs_orig_abs_cm": mj(jf, jo), "full_vs_orig_rootaligned_cm": mj(jf, jo, True),
                "full_vs_precise_clip_abs_cm": mj(jf, jp), "full_vs_precise_clip_rootaligned_cm": mj(jf, jp, True),
                "precise_clip_vs_orig_abs_cm": mj(jp, jo), "precise_clip_vs_orig_rootaligned_cm": mj(jp, jo, True),
                "full_vs_precise_clip_pose_angle_deg_mean_p95": ang(full["smpl_cam_body_pose"][sl][:T][m], rp["smpl_cam_body_pose"][:T][m]),
                "full_vs_precise_clip_global_orient_deg": float(np.degrees((R.from_rotvec(full["smpl_cam_global_orient"][sl][:T][m].astype(np.float64)).inv()
                                                                              * R.from_rotvec(rp["smpl_cam_global_orient"][:T][m].astype(np.float64))).magnitude()).mean()),
                "vo_failed_full_segments_in_span": int(np.unique(full["segment_id"][sl][pres][full["cam_vo_failed"][sl][pres]]).size),
                "vo_failed_clip": int(rp["segments"][:, 2].sum()),
            })
            rows.append(row)
            print(json.dumps(row), flush=True)
            (out / "clip_compare.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if "frames_compared" in r]
    if ok:
        summ = {k: round(float(np.mean([r[k] for r in ok])), 3) for k in ok[0] if k.endswith("_cm") or k.endswith("_deg")}
        print("MEAN", json.dumps(summ))
        (out / "clip_compare_summary.json").write_text(json.dumps({"n_clips": len(ok), "mean": summ}, indent=1))


if __name__ == "__main__":
    main()
