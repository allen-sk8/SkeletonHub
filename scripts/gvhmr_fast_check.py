"""gvhmr_fast vs original GVHMR on hmr_zoo clip dirs (run in skeleton_env, one GPU, CUDA_VISIBLE_DEVICES set outside).

  python scripts/gvhmr_fast_check.py --clips outputs/hmr_zoo/figure_skating/3A_3A_0133,... --out outputs/finefs_full_test/check

For each clip, with the clip's own boxes (so the box logic is identical):
  orig   = <clip>/gvhmr.npz written earlier by hmr_zoo (the original GVHMR detector, fp32 + flip test, SimpleVO)
  precise= gvhmr_fast code path with fp32 + flip test (same code, original numerics)
  fast   = gvhmr_fast (fp16, no flip test)
Metrics (camera space, H36M-17 joints, metres->cm): mean joint distance in absolute camera coordinates and after
subtracting the pelvis; and (precise vs fast) body-pose geodesic angle per joint (deg).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path = [p for p in sys.path if ".local" not in p] + [p for p in sys.path if ".local" in p]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402


def mj(a, b, root_align=False):
    if root_align:
        a, b = a - a[:, :1], b - b[:, :1]
    return float(np.linalg.norm(a - b, axis=-1).mean() * 100)


def pose_angle_deg(p, q):
    """mean geodesic angle between two (T,63) axis-angle body poses over 21 joints (+ the 2 hands are included: 23)."""
    T = p.shape[0]
    Rp = R.from_rotvec(p.reshape(-1, 3).astype(np.float64))
    Rq = R.from_rotvec(q.reshape(-1, 3).astype(np.float64))
    ang = np.degrees((Rp.inv() * Rq).magnitude()).reshape(T, 21)
    return float(ang.mean()), float(np.percentile(ang, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import os

    a.out = str(Path(a.out).resolve())
    a.clips = ",".join(str((Path(c) if Path(c).is_absolute() else Path.cwd() / c).resolve()) for c in a.clips.split(","))
    os.chdir(str(ROOT / "external" / "gvhmr"))
    from detectors.gvhmr_fast_detector import GVHMRFastDetector, save_fast_npz

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = GVHMRFastDetector(a.device, fp16=True, flip_test=False)
    rows = []
    for c in a.clips.split(","):
        clip = Path(c) if Path(c).is_absolute() else ROOT / c
        key = clip.name
        mp4, bj = clip / "clip.mp4", clip / "boxes.json"
        orig = np.load(clip / "gvhmr.npz")
        res = {}
        for tag, fp16, flip in (("fast", True, False), ("precise", False, True)):
            det.set_mode(fp16, flip)
            det.detect_video_fast(mp4, bj, verbose=False)        # warm-up (CUDA init, cudnn autotune), discarded
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = det.detect_video_fast(mp4, bj, verbose=False)
            torch.cuda.synchronize()
            r["_sec"] = time.perf_counter() - t0
            res[tag] = r
            save_fast_npz(out / f"{key}_{tag}.npz", {k: v for k, v in r.items() if k != "_sec"})
        f, p = res["fast"], res["precise"]
        T = f["joints17_cam"].shape[0]
        mf = json.loads(str(f["meta_json"]))
        mp = json.loads(str(p["meta_json"]))
        row = {
            "clip": key, "frames": T,
            "fast_vs_orig_abs_cm": mj(f["joints17_cam"], orig["joints3d_cam"]),
            "fast_vs_orig_rootaligned_cm": mj(f["joints17_cam"], orig["joints3d_cam"], True),
            "precise_vs_orig_abs_cm": mj(p["joints17_cam"], orig["joints3d_cam"]),
            "precise_vs_orig_rootaligned_cm": mj(p["joints17_cam"], orig["joints3d_cam"], True),
            "fast_vs_precise_abs_cm": mj(f["joints17_cam"], p["joints17_cam"]),
            "fast_vs_precise_rootaligned_cm": mj(f["joints17_cam"], p["joints17_cam"], True),
            "fast_vs_precise_world_rootaligned_cm": mj(f["joints24_world"], p["joints24_world"], True),
            "orig_vs_orig_shift_note": "",
            "pose_angle_deg_mean_p95": pose_angle_deg(f["smpl_cam_body_pose"], p["smpl_cam_body_pose"]),
            "global_orient_angle_deg": float(np.degrees((R.from_rotvec(f["smpl_cam_global_orient"].astype(np.float64)).inv()
                                                          * R.from_rotvec(p["smpl_cam_global_orient"].astype(np.float64))).magnitude()).mean()),
            "fast_sec": round(f["_sec"], 2), "precise_sec": round(p["_sec"], 2),
            "fast_timing": mf["timing_sec"], "precise_timing": mp["timing_sec"],
            "fast_vram_gb": mf["torch_peak_gb"], "precise_vram_gb": mp["torch_peak_gb"],
            "vo_failed": [int(f["segments"][:, 2].sum()), int(p["segments"][:, 2].sum())],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=1), flush=True)
    (out / "check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
