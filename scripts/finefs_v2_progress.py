"""Progress file for the FineFS v2 full batch: writes <work>/PROGRESS.md (read daily). CPU only, a few seconds.

  python scripts/finefs_v2_progress.py --work outputs/finefs_v2_full [--root /home/allen/datasets/FineFS]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", default="/home/allen/datasets/FineFS")
    a = ap.parse_args()
    W, R = Path(a.work), Path(a.root)
    ids = [l.strip() for l in (W / "all_ids.txt").read_text().split() if l.strip()]
    done, done_t = [], []
    for v in ids:
        p, g = R / "5_preview" / f"{v}.mp4", R / "3_gvhmr" / v / "gvhmr_fast.npz"
        if p.exists() and g.exists() and (R / "4_flags" / v / "flags.npz").exists():
            done.append(v)
            done_t.append(p.stat().st_mtime)
    no_person, tracked = [], 0
    for v in ids:
        st = W / "_track" / v / "track_stats.json"
        if st.exists() or (R / "1_tracking" / v / "boxes.json").exists():
            tracked += 1
        if st.exists() and json.loads(st.read_text()).get("no_person"):
            no_person.append(v)
    failed = {}
    for f in sorted((W / "_failed").glob("*.txt")) if (W / "_failed").exists() else []:
        v, stage = f.stem.split(".", 1)
        failed[v] = f"{stage}: {f.read_text().splitlines()[0]}"
    now = time.time()
    left = len(ids) - len(done) - len(no_person) - len(failed)
    t0 = float((W / "start_time").read_text()) if (W / "start_time").exists() else now
    since = max(t0, now - 24 * 3600)                           # rate over the last day of this batch (not older test runs)
    recent = [t for t in done_t if t > since]
    if len(recent) >= 5:
        rate = len(recent) / ((now - since) / 3600)
        eta = f"about {left / rate / 24:.1f} days ({rate:.1f} videos per hour, measured since {time.strftime('%m-%d %H:%M', time.localtime(since))})"
    else:
        eta = "not enough finished videos in the last day to estimate"
    state = "running"
    stop = W / "STOP"
    if (W / "ALL_DONE").exists():
        state = "ALL DONE"
    elif stop.exists():
        state = f"stopped: {stop.read_text().strip()}"
    sup = W / "supervisor.last"
    lines = [f"# FineFS v2 full batch progress (updated {time.strftime('%F %T')})", "",
             f"- State: {state}" + (f"; supervisor: {sup.read_text().strip()}" if sup.exists() else ""),
             f"- Finished: {len(done)} of {len(ids)} videos (all stages, in {R})",
             f"- Tracked so far: {tracked}",
             f"- Failed: {len(failed)}" + (" (" + ", ".join(f"{v} {s}" for v, s in list(failed.items())[:20]) + ")" if failed else ""),
             f"- No skater found (skipped on purpose): {len(no_person)}" + (f" ({', '.join(no_person[:20])})" if no_person else ""),
             f"- Left: {left}; estimated time left: {eta}",
             f"- Log: {W}/batch.log, failure details: {W}/_failed/", ""]
    (W / "PROGRESS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
