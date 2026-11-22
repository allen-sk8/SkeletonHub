"""Aggregate stage A/B stats of the whole-match test into tables + extrapolation to the full 1,167 videos (stdlib only).
  python scripts/finefs_full_summary.py --out outputs/finefs_full_test [--n-videos 1167 --total-video-hours 65.3]
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-videos", type=int, default=1167)
    ap.add_argument("--total-video-hours", type=float, default=65.31)
    a = ap.parse_args()
    out = Path(a.out)
    probe = {p["id"]: p for p in json.loads((out / "video_probe.json").read_text())}
    rows = []
    for d in sorted(out.iterdir()):
        if not (d / "track_stats.json").exists():
            continue
        t = json.loads((d / "track_stats.json").read_text())
        f = json.loads((d / "fast_stats.json").read_text()) if (d / "fast_stats.json").exists() else {}
        p = probe[t["id"]]
        rows.append((t, f, p))
    print("| video | size | video sec | detect x | track x | lock x | GVHMR-fast x | fps(run) | present | cuts | segs | VO fail | VRAM GB | npz MB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    S = dict(v=0.0, det=0.0, trk=0.0, lock=0.0, fast=0.0, mb=0.0)
    for t, f, p in rows:
        v = t["video_sec"]
        fx = f.get("wall_sec", 0) / v if f else float("nan")
        print(f"| {t['id']} | {p['w']}x{p['h']} | {v:.0f} | {t['detect_sec']/v:.2f} | {t['track_sec']/v:.2f} | {t['lock_sec']/v:.2f} | {fx:.2f} | "
              f"{f.get('fps_processed','-')} | {t['present_ratio']:.3f} | {t['n_shot_cuts']} | {f.get('segments','-')} | {f.get('vo_failed_segments','-')} | "
              f"{max(f.get('nvsmi_peak_gb',0), f.get('torch_peak_gb',0)) if f else '-'} | {f.get('npz_mb','-')} |")
        S["v"] += v; S["det"] += t["detect_sec"]; S["trk"] += t["track_sec"]; S["lock"] += t["lock_sec"]
        S["fast"] += f.get("wall_sec", 0); S["mb"] += f.get("npz_mb", 0)
    v = S["v"]
    r_det, r_trk, r_lock, r_fast = S["det"] / v, S["trk"] / v, S["lock"] / v, S["fast"] / v
    print(f"\nSum video {v/60:.1f} min over {len(rows)} videos. Ratios to video length: detect {r_det:.2f}, track {r_trk:.2f}, lock {r_lock:.2f}, gvhmr_fast {r_fast:.2f}, total {r_det+r_trk+r_lock+r_fast:.2f}")
    H = a.total_video_hours
    tot = (r_det + r_trk + r_lock + r_fast) * H
    mb = S["mb"] / len(rows) * a.n_videos / 1e3
    print(f"Extrapolation to {a.n_videos} videos ({H} h of video): {tot:.0f} GPU-hours total  -> 1 card {tot/24:.1f} days, 3 cards {tot/72:.1f} days; npz {mb:.1f} GB (+ boxes/tracks ~ see doc)")
    (out / "summary_extrapolation.json").write_text(json.dumps({"ratios": dict(detect=r_det, track=r_trk, lock=r_lock, fast=r_fast), "gpu_hours": tot, "npz_gb": mb}, indent=1))


if __name__ == "__main__":
    main()
