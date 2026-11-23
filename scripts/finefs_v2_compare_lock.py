"""Compare two lock-on results (boxes.json) of the same whole-match videos: missing frames, frames missing right after
each camera cut, and where the two disagree (frames to look at by eye).

  python scripts/finefs_v2_compare_lock.py --a outputs/finefs_full_test --b outputs/finefs_v2_test/_exp1_offline \
      --ids 36,261 --out outputs/finefs_v2_test/_exp1_offline/compare.json [--sheet-dir DIR]

"after cut" = for every shot (frames between two cuts) in which at least one of the two results has the person, the number of
frames from the shot start to that result's first present frame (shot length if never present).
Disagreement runs: frames where exactly one is present, or both are present with box IoU < 0.5 (tight boxes).
With --sheet-dir, a contact sheet per video shows the middle frame of the longest disagreement runs, A box red, B box green.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def runs(mask):
    out, t, n = [], 0, len(mask)
    while t < n:
        if mask[t]:
            s = t
            while t < n and mask[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def compare(a, b):
    n = int(a["n_frames"])
    pa, pb = np.asarray(a["present"], bool)[:n], np.asarray(b["present"], bool)[:n]
    ta, tb = a.get("tight_clean"), b.get("tight_clean")
    cuts = sorted({int(c[0]) for c in a.get("shot_cuts", [])})
    edges = [0] + [c for c in cuts if 0 < c < n] + [n]
    after_a, after_b = [], []
    for s, e in zip(edges[:-1], edges[1:]):
        if not (pa[s:e].any() or pb[s:e].any()):
            continue
        fa = int(np.argmax(pa[s:e])) if pa[s:e].any() else e - s
        fb = int(np.argmax(pb[s:e])) if pb[s:e].any() else e - s
        if s > 0:          # the first shot starts at frame 0, not after a cut
            after_a.append(fa)
            after_b.append(fb)
    both = pa & pb
    ious = np.array([iou(ta[t], tb[t]) for t in np.where(both)[0]]) if both.any() else np.zeros(0)
    low = np.zeros(n, bool)
    low[np.where(both)[0][ious < 0.5]] = True
    dis = (pa ^ pb) | low
    rr = sorted(runs(dis), key=lambda r: -(r[1] - r[0]))
    return {"n": n, "missing_a": round(1 - pa.mean(), 4), "missing_b": round(1 - pb.mean(), 4),
            "only_a": int((pa & ~pb).sum()), "only_b": int((pb & ~pa).sum()),
            "both_iou_lt_0.5": int(low.sum()), "both_iou_mean": round(float(ious.mean()), 4) if len(ious) else None,
            "shots_with_person": len(after_a), "after_cut_a_mean": round(float(np.mean(after_a)), 2) if after_a else None,
            "after_cut_b_mean": round(float(np.mean(after_b)), 2) if after_b else None,
            "after_cut_a": after_a, "after_cut_b": after_b,
            "after_cut_a_le0": int(sum(x == 0 for x in after_a)), "after_cut_b_le0": int(sum(x == 0 for x in after_b)),
            "disagree_runs": [[int(s), int(e), "only_a" if pa[s] and not pb[s] else ("only_b" if pb[s] and not pa[s] else "iou")]
                              for s, e in rr[:40]]}


def sheet(video, a, b, res, out_png, k=12):
    rr = [r for r in res["disagree_runs"] if r[1] - r[0] >= 3][:k]
    if not rr:
        return 0
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for s, e, kind in rr:
        f = (s + e) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        H = fr.shape[0]
        for res_, col in ((a, (0, 0, 255)), (b, (60, 245, 70))):
            if res_["present"][f]:
                x = res_["tight_clean"][f]
                cv2.rectangle(fr, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), col, max(3, H // 180))
        cv2.putText(fr, f"{kind} f{s}-{e} ({s/25:.1f}s, {e-s}f)", (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, H / 450, (255, 255, 0), 3)
        tiles.append(cv2.resize(fr, (480, int(480 * H / fr.shape[1]))))
    cap.release()
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out_png), np.concatenate(rows, 0))
    return len(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dir with <id>/boxes.json (reference, red)")
    ap.add_argument("--b", required=True, help="dir with <id>/boxes.json (new, green)")
    ap.add_argument("--ids", default="36,261,819,739,980,652,187,513,521,456")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet-dir")
    ap.add_argument("--video-dir", default="/home/allen/datasets/FineFS/video")
    a = ap.parse_args()
    allres = {}
    for vid in a.ids.split(","):
        ja = json.loads((Path(a.a) / vid / "boxes.json").read_text())
        jb = json.loads((Path(a.b) / vid / "boxes.json").read_text())
        r = compare(ja, jb)
        if a.sheet_dir:
            Path(a.sheet_dir).mkdir(parents=True, exist_ok=True)
            r["sheet_tiles"] = sheet(Path(a.video_dir) / f"{vid}.mp4", ja, jb, r, Path(a.sheet_dir) / f"disagree_{vid}.png")
        allres[vid] = r
        print(vid, {k: v for k, v in r.items() if k not in ("after_cut_a", "after_cut_b", "disagree_runs")})
    Path(a.out).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
