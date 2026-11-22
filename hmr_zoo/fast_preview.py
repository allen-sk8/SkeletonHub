"""Preview video for a gvhmr_fast result: tracked box + SMPL-24 skeleton (camera-space joints projected with K) over the
original video, with segment id, shot cuts and 'NO PERSON' marks. Run with any python that has cv2 + numpy + scipy.

  python -m hmr_zoo.fast_preview --video V.mp4 --boxes boxes.json --npz gvhmr_fast.npz --start 20 --dur 45 --out prev.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hmr_zoo.fast_io import load_fast  # noqa: E402

SMPL24_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
COLORS = [(0, 0, 255), (255, 80, 0), (0, 220, 0), (0, 0, 255), (255, 80, 0)]


def render(video, boxes_json, npz, start, dur, out, width=960):
    d = load_fast(npz)
    bj = json.loads(Path(boxes_json).read_text())
    fps = float(d["fps"])
    K = d["K"]
    cuts = {int(c[0]) for c in bj.get("shot_cuts", [])}
    cap = cv2.VideoCapture(str(video))
    f0 = int(start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = min(1.0, width / W)
    size = (int(W * sc) // 2 * 2, int(H * sc) // 2 * 2)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                          "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", str(out)],
                         stdin=subprocess.PIPE)
    pad = bj.get("padded_112")
    th = max(2, H // 300)
    for t in range(f0, min(f0 + int(dur * fps), len(d["present"]))):
        ok, fr = cap.read()
        if not ok:
            break
        if d["present"][t]:
            j = d["joints24_cam"][t]
            z = np.maximum(j[:, 2], 0.05)
            uv = np.stack([K[0, 0] * j[:, 0] / z + K[0, 2], K[1, 1] * j[:, 1] / z + K[1, 2]], 1)
            for ch, col in zip(SMPL24_CHAIN, COLORS):
                for a, b in zip(ch[:-1], ch[1:]):
                    cv2.line(fr, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), col, th + 1, cv2.LINE_AA)
            if pad is not None:
                b = pad[t]
                cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 245, 70), th)
            tag = f"seg {int(d['segment_id'][t])}" + ("  (static cam: VO failed)" if d["cam_vo_failed"][t] else "")
            cv2.putText(fr, tag, (20, int(H * 0.06)), cv2.FONT_HERSHEY_SIMPLEX, H / 700, (60, 245, 70), 2, cv2.LINE_AA)
        else:
            cv2.putText(fr, "NO PERSON (not run)", (20, int(H * 0.06)), cv2.FONT_HERSHEY_SIMPLEX, H / 700, (0, 0, 255), 3, cv2.LINE_AA)
        if t in cuts:
            cv2.rectangle(fr, (0, 0), (W - 1, H - 1), (0, 255, 255), th * 4)
            cv2.putText(fr, "SHOT CUT", (20, int(H * 0.12)), cv2.FONT_HERSHEY_SIMPLEX, H / 700, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(fr, f"{t / fps:6.1f}s", (W - int(W * 0.12), int(H * 0.06)), cv2.FONT_HERSHEY_SIMPLEX, H / 700, (255, 255, 255), 2, cv2.LINE_AA)
        p.stdin.write(cv2.resize(fr, size).tobytes())
    cap.release()
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=45)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    render(a.video, a.boxes, a.npz, a.start, a.dur, a.out)
