"""Side-by-side comparison videos for the FineFS v2 test (CPU, cv2 + ffmpeg).

  python scripts/finefs_v2_compare_video.py track   # old lock (finefs_full_test) | final lock (finefs_v2_test/1_tracking)
  python scripts/finefs_v2_compare_video.py res     # crop from 0.5x frames | crop from full-resolution frames
Writes outputs/finefs_v2_test/compare_videos/{tracking_old_vs_new.mp4, resolution_half_vs_full.mp4}.

track: chosen stretches where the two lock results disagree (from _compare_old_vs_final.json), +-1 s around each,
       at most 6 s each. Left = old box (red), right = new box (green), 'NO PERSON' where absent.
res:   per video, the 3 stretches (3 s) where the half / full 3D joints differ most. Both panels show the same crop of the
       ORIGINAL frame around the skater; drawn on it: ViTPose 2D of that variant (cyan, conf >= 0.3) and the SMPL joints of
       that variant's GVHMR result projected with K (magenta).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "outputs" / "finefs_v2_test"
VIDEO_DIR = Path("/home/allen/datasets/FineFS/video")
OUT = V / "compare_videos"
COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 1), (0, 2), (1, 3), (2, 4)]
SMPL_EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 15),
              (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]

TRACK_SEGS = [("456", 751, 779, "old: spectator, new: skater"), ("456", 1456, 1506, "old: spectator, new: skater"),
              ("456", 2160, 2301, "old: no person, new: skater"), ("456", 3123, 3200, "boxes differ"),
              ("261", 1371, 1456, "old: no person, new: skater"), ("261", 2868, 3004, "old: no person, new: skater"),
              ("819", 2060, 2199, "old: no person, new: skater"), ("980", 5427, 5446, "old: no person, new: skater"),
              ("187", 0, 31, "close-up face at the start")]
RES_IDS = ["261", "819", "187"]


def text(img, s, y, scale=0.6, col=(255, 255, 255)):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1)


def writer(path, size, fps=25):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                             str(path)], stdin=subprocess.PIPE)


def read_range(video, a, b):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    for t in range(a, b):
        ok, fr = cap.read()
        if not ok:
            break
        yield t, fr
    cap.release()


def track():
    PW = 640
    p = None
    for vid, s, e, note in TRACK_SEGS:
        old = json.loads((ROOT / "outputs/finefs_full_test" / vid / "boxes.json").read_text())
        new = json.loads((V / "1_tracking" / vid / "boxes.json").read_text())
        n = int(new["n_frames"])
        a, b = max(0, s - 25), min(n, max(e, s + 50) + 25)
        b = min(b, a + 150)
        for t, fr in read_range(VIDEO_DIR / f"{vid}.mp4", a, b):
            H, W = fr.shape[:2]
            sc = PW / W
            panels = []
            for res, col, name in ((old, (0, 0, 255), "OLD: online lock, tracker without Re-ID"),
                                   (new, (60, 245, 70), "NEW: tracker Re-ID + offline lock")):
                im = cv2.resize(fr, (PW, int(round(H * sc / 2)) * 2), interpolation=cv2.INTER_AREA)
                if res["present"][t]:
                    x = np.asarray(res["tight_clean"][t]) * sc
                    cv2.rectangle(im, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), col, 3)
                else:
                    text(im, "NO PERSON", 60, 1.0, (0, 0, 255))
                text(im, name, 22, 0.55, col)
                text(im, f"video {vid}  frame {t}  {t / 25:.1f}s   [{note}]", im.shape[0] - 12, 0.5)
                panels.append(im)
            frame = np.concatenate(panels, 1)
            if p is None:
                size = (frame.shape[1], frame.shape[0])
                p = writer(OUT / "tracking_old_vs_new.mp4", size)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            p.stdin.write(frame.tobytes())
    p.stdin.close()
    p.wait()


def project(j, K):
    u = j @ K.T
    return u[:, :2] / np.maximum(u[:, 2:3], 1e-6)


def res():
    PS = 480
    p = writer(OUT / "resolution_half_vs_full.mp4", (2 * PS, PS))
    for vid in RES_IDS:
        gh = np.load(V / "3_gvhmr" / vid / "gvhmr_fast.npz")
        gf = np.load(V / "_exp3_resolution" / vid / "gvhmr_full.npz")
        kh = np.load(V / "2_pose2d" / vid / "vitpose_h.npz")["kp2d"]
        kf = np.load(V / "_exp3_resolution" / vid / "pose2d_full.npz")["kp2d"]
        bj = json.loads((V / "1_tracking" / vid / "boxes.json").read_text())
        pad = np.asarray(bj["padded_125"], np.float64)
        K = gh["K"]
        jh, jf = gh["joints24_cam"], gf["joints24_cam"]
        d = np.linalg.norm((jh - jh[:, :1]) - (jf - jf[:, :1]), axis=-1).mean(1)
        d = np.where(np.isfinite(d), d, 0)
        dsm = np.convolve(d, np.ones(75) / 75, mode="same")
        picks = []
        for t in np.argsort(-dsm):
            if all(abs(t - q) > 150 for q in picks) and gh["present"][max(0, t - 37):t + 38].all():
                picks.append(int(t))
            if len(picks) == 3:
                break
        for c in sorted(picks):
            a, b = c - 37, c + 38
            for t, fr in read_range(VIDEO_DIR / f"{vid}.mp4", a, b):
                x1, y1, x2, y2 = pad[t]
                cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
                A = np.array([[PS / s, 0, PS / 2 - cx * PS / s], [0, PS / s, PS / 2 - cy * PS / s]])
                panels = []
                for name, kp, g in (("HALF: crop from 0.5x frame (now)", kh, gh), ("FULL: crop from full-res frame", kf, gf)):
                    im = cv2.warpAffine(fr, A, (PS, PS))
                    k = kp[t].copy()
                    k[:, :2] = k[:, :2] @ A[:, :2].T + A[:, 2]
                    for i, j in COCO_EDGES:
                        if k[i, 2] >= 0.3 and k[j, 2] >= 0.3:
                            cv2.line(im, tuple(k[i, :2].astype(int)), tuple(k[j, :2].astype(int)), (255, 220, 0), 2)
                    u = project(g["joints24_cam"][t], K) @ A[:, :2].T + A[:, 2]
                    for i, j in SMPL_EDGES:
                        cv2.line(im, tuple(u[i].astype(int)), tuple(u[j].astype(int)), (255, 0, 255), 1)
                    text(im, name, 20, 0.5)
                    text(im, f"2D conf {np.nanmean(kp[t, :, 2]):.2f}", 42, 0.5, (255, 220, 0))
                    text(im, f"video {vid} {int(bj['height'])}p  frame {t}  {t / 25:.1f}s  diff {d[t] * 100:.1f}cm", PS - 12, 0.45)
                    panels.append(im)
                text(panels[1], "cyan = ViTPose 2D, magenta = GVHMR 3D projected", 64, 0.45)
                p.stdin.write(np.concatenate(panels, 1).tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    {"track": track, "res": res}[sys.argv[1]]()
