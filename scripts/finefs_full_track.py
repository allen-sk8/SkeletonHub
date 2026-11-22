"""Stage A for whole-match FineFS videos (run in the reidtrack env, one GPU): lock-on person tracking -> boxes.json.

  python scripts/finefs_full_track.py --ids 36,261 --out outputs/finefs_full_test [--video-dir /home/allen/datasets/FineFS/video]

Per video writes <out>/<id>/{boxes.json, boxes_padded_112.txt, tracks.txt, detections_medium.npz, track_stats.json}.
Existing boxes.json are skipped (resumable). Tracking settings are preprocessors/track_person.py defaults
(RF-DETR Medium + BoT-SORT + OSNet Re-ID, --target lock, preview off; previews are made separately for a few videos).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default="/home/allen/datasets/FineFS/video")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-gb", type=float, default=6.0)
    ap.add_argument("--reuse-tracks", action="store_true", help="redo only the lock/presence step on saved tracks.txt (ignores existing boxes.json)")
    ap.add_argument("--tracker", default="botsort-reid", choices=["botsort", "botsort-reid"])
    a = ap.parse_args()
    import torch
    from preprocessors.track_person import track_video

    for vid in [v for v in a.ids.split(",") if v]:
        d = Path(a.out) / vid
        if (d / "boxes.json").exists() and not a.reuse_tracks:
            print(f"[track] {vid}: exists, skipped", flush=True)
            continue
        video = Path(a.video_dir) / f"{vid}.mp4"
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        p = track_video(video, d, detector="medium", tracker=a.tracker, target="lock", device=a.device,
                        gpu_mem_gb=a.gpu_mem_gb, preview=False, reuse_tracks=a.reuse_tracks)
        wall = time.perf_counter() - t0
        n = p["n_frames"]
        st = {"id": vid, "frames": n, "fps": p["fps"], "video_sec": n / p["fps"], "wall_sec": round(wall, 1),
              "detect_sec": p.get("det_seconds"), "track_sec": p.get("track_seconds"), "lock_sec": p.get("lock_seconds"),
              "wall_over_video": round(wall / (n / p["fps"]), 2),
              "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
              "present_ratio": round(sum(p["present"]) / n, 4), "n_shot_cuts": len(p.get("shot_cuts", [])),
              "no_person": bool(p.get("no_person"))}
        (d / "track_stats.json").write_text(json.dumps(st, indent=1))
        print("[track]", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
