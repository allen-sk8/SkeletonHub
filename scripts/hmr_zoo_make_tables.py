"""Collect outputs/hmr_zoo/<sport>/<clip>/metrics.csv into markdown tables (per sport, and per-model sport means).

    python scripts/hmr_zoo_make_tables.py [--root outputs/hmr_zoo] > tables.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

COLS = [("reproj_in_box_pct", "in-box %"), ("center_err_pct_h", "ctr err %h"), ("proj_h_ratio", "h ratio"),
        ("accel_jitter_mps2", "jitter"), ("bone_len_cv_pct", "boneCV %"), ("pelvis_yaw_p95_deg", "yaw p95"),
        ("depth_z_jump_p95_m", "z-jump m"), ("fps_infer", "fps"), ("peak_vram_gb", "VRAM GB")]


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "outputs/hmr_zoo"))
    a = ap.parse_args()
    root = Path(a.root)
    rows = []
    for p in sorted(root.glob("*/*/metrics.csv")):
        rows += list(csv.DictReader(open(p)))
    by_sport = defaultdict(list)
    for r in rows:
        by_sport[r["sport"]].append(r)
    for sport, rs in by_sport.items():
        print(f"\n### {sport}\n")
        for clip in sorted({r["clip"] for r in rs}):
            cr = [r for r in rs if r["clip"] == clip]
            print(f"**{clip}** ({cr[0]['frames']} frames, {cr[0]['width']}x{cr[0]['height']})\n")
            print("| model | " + " | ".join(h for _, h in COLS) + " |")
            print("|---|" + "---|" * len(COLS))
            for r in cr:
                print(f"| {r['model']} | " + " | ".join(f"{float(r[k]):.2f}" for k, _ in COLS) + " |")
            print()
        if len({r["clip"] for r in rs}) > 1:
            print(f"**{sport}: mean over clips**\n")
            print("| model | " + " | ".join(h for _, h in COLS) + " |")
            print("|---|" + "---|" * len(COLS))
            for m in dict.fromkeys(r["model"] for r in rs):
                mr = [r for r in rs if r["model"] == m]
                print(f"| {m} | " + " | ".join(f"{mean([float(r[k]) for r in mr]):.2f}" for k, _ in COLS) + " |")
            print()


if __name__ == "__main__":
    main()
