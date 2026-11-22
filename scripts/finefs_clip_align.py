"""Find where each FineFS_5s clip sits inside its whole-match video (start frame), by thumbnail matching.
  python scripts/finefs_clip_align.py --ids 36,261 --out outputs/finefs_full_test
Writes <out>/<id>/clips.json: [{element, clip, time, start_frame, n_frames, match_err}]
(clip 'time' is 'min-sec,min-sec' of the element in the match; the 5 s clip is cut around it.)
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np

IDX = Path("/home/allen/SkeletonHub/outputs/finefs_full_test/clip_index.json")


def thumbs(path, f0=0, n=None):
    cap = cv2.VideoCapture(str(path))
    if f0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out, i = [], 0
    while n is None or i < n:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (48, 27), interpolation=cv2.INTER_AREA).astype(np.float32))
        i += 1
    cap.release()
    return np.stack(out) if out else np.zeros((0, 27, 48), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default="/home/allen/datasets/FineFS/video")
    a = ap.parse_args()
    idx = json.loads(IDX.read_text())
    for vid in a.ids.split(","):
        video = Path(a.video_dir) / f"{vid}.mp4"
        res = []
        for element, clip, tm, info in idx[vid]:
            s = tm.split(",")[0].split("-")
            t0 = int(s[0]) * 60 + int(s[1])
            clip_mp4 = Path(info).with_name(f"{clip}.mp4")
            ct = thumbs(clip_mp4)
            win0 = max(0, int((t0 - 8) * 25))
            vt = thumbs(video, win0, int(24 * 25))
            best, besti = 1e18, -1
            for i in range(0, len(vt) - 20):
                e = float(np.mean((vt[i:i + 10] - ct[:10]) ** 2))
                if e < best:
                    best, besti = e, i
            res.append({"element": element, "clip": clip, "time": tm, "start_frame": win0 + besti, "n_frames": len(ct),
                        "match_err": round(best, 2), "clip_mp4": str(clip_mp4)})
            print(vid, element, clip, tm, "start_frame", win0 + besti, "t0*25", t0 * 25, "err", round(best, 2), flush=True)
        d = Path(a.out) / vid
        d.mkdir(parents=True, exist_ok=True)
        (d / "clips.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
