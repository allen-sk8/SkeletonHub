"""Stage B for whole-match FineFS videos (run in skeleton_env, one GPU): gvhmr_fast on the locked boxes -> npz.

  python scripts/finefs_full_fast.py --ids 36,261 --out outputs/finefs_full_test
Reads <out>/<id>/boxes.json (from finefs_full_track.py), writes <out>/<id>/gvhmr_fast.npz and fast_stats.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path = [p for p in sys.path if ".local" not in p] + [p for p in sys.path if ".local" in p]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default="/home/allen/datasets/FineFS/video")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    os.chdir(str(ROOT / "external" / "gvhmr"))
    import torch
    from detectors.gvhmr_fast_detector import GVHMRFastDetector, save_fast_npz
    from hmr_zoo.vram import ProcessVramMonitor

    base = ProcessVramMonitor().baseline_mb
    det = GVHMRFastDetector(a.device, fp16=True, flip_test=False)
    for vid in [v for v in a.ids.split(",") if v]:
        d = out / vid
        if (d / "gvhmr_fast.npz").exists():
            print(f"[fast] {vid}: exists, skipped", flush=True)
            continue
        bj = json.loads((d / "boxes.json").read_text())
        if bj.get("no_person"):
            print(f"[fast] {vid}: no person locked, nothing to run", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with ProcessVramMonitor(baseline_mb=base) as mon:
            res = det.detect_video_fast(Path(a.video_dir) / f"{vid}.mp4", d / "boxes.json", verbose=False)
        wall = time.perf_counter() - t0
        save_fast_npz(d / "gvhmr_fast.npz", res)
        meta = json.loads(str(res["meta_json"]))
        n = meta["n_frames"]
        st = {"id": vid, "frames": n, "video_sec": n / float(res["fps"]), "wall_sec": round(wall, 1),
              "wall_over_video": round(wall / (n / float(res["fps"])), 2), "fps_processed": round(meta["n_run"] / wall, 1),
              "n_run": meta["n_run"], "present_ratio_run": round(meta["n_run"] / n, 4), "segments": meta["n_segments"],
              "vo_failed_segments": meta["n_vo_failed_segments"], "timing_sec": meta["timing_sec"],
              "torch_peak_gb": meta["torch_peak_gb"], "nvsmi_peak_gb": round(mon.peak_gb, 2),
              "npz_mb": round((d / "gvhmr_fast.npz").stat().st_size / 1e6, 2)}
        (d / "fast_stats.json").write_text(json.dumps(st, indent=1))
        print("[fast]", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
