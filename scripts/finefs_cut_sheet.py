"""Contact sheet around shot cuts and present/absent transitions of one tracked video (CPU, cv2).
  python scripts/finefs_cut_sheet.py --id 36 --out outputs/finefs_full_test [--max 10]
Each row: frame before | frame after, with the locked box (green) / 'NO PERSON' and frame numbers. Saved as <out>/<id>/cut_sheet.png
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--video-dir", default="/home/allen/datasets/FineFS/video")
    a = ap.parse_args()
    d = Path(a.out) / a.id
    bj = json.loads((d / "boxes.json").read_text())
    n = bj["n_frames"]
    present = np.asarray(bj["present"], bool)
    pad = bj.get("padded_112")
    cuts = [int(c[0]) for c in bj.get("shot_cuts", [])]
    trans = [t for t in range(1, n) if present[t] != present[t - 1]]
    ev = [("cut", c) for c in cuts] + [("pres", t) for t in trans]
    ev = sorted(ev, key=lambda e: e[1])
    if len(ev) > a.max:
        ev = [ev[i] for i in np.linspace(0, len(ev) - 1, a.max).round().astype(int)]
    cap = cv2.VideoCapture(str(Path(a.video_dir) / f"{a.id}.mp4"))
    rows = []
    for kind, f in ev:
        pair = []
        for g in (max(f - 2, 0), min(f + 1, n - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, g)
            ok, fr = cap.read()
            if not ok:
                fr = np.zeros((bj["height"], bj["width"], 3), np.uint8)
            H, W = fr.shape[:2]
            if present[g] and pad:
                b = pad[g]
                cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 245, 70), max(3, H // 200))
            else:
                cv2.putText(fr, "NO PERSON", (30, int(H * .12)), cv2.FONT_HERSHEY_SIMPLEX, H / 300, (0, 0, 255), 4)
            cv2.putText(fr, f"{kind} frame {g} ({g/25:.1f}s)", (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, H / 500, (255, 255, 0), 3)
            pair.append(cv2.resize(fr, (480, int(480 * H / W))))
        rows.append(np.concatenate(pair, 1))
    if rows:
        cv2.imwrite(str(d / "cut_sheet.png"), np.concatenate(rows, 0))
    print(a.id, "events", len(ev), "of cuts", len(cuts), "transitions", len(trans))


if __name__ == "__main__":
    main()
