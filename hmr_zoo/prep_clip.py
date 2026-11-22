"""Make a clip directory: cut a video segment and compute the shared person boxes.

Run in the `reidtrack` conda env. Tracking is preprocessors/track_person.py (RF-DETR Medium + BoT-SORT with osnet
Re-ID, same settings as ~/reid_tracking/scripts/fs_track.py "method D"; finefs box repair + 1.12x padding).

  # new video segment (tracking = preprocessors/track_person.py) (5-10 s)
  python -m hmr_zoo.prep_clip --video SRC.mp4 --start 14.9 --dur 7 --sport roller_skating --name v25_3F_14.9s --out OUTDIR
  # reuse the boxes of a finefs_preproc Stage 2 clip (copies clip + boxes.json)
  python -m hmr_zoo.prep_clip --finefs 3A_3A_0133 --sport figure_skating --out OUTDIR

OUTDIR/<sport>/<name>/ gets clip.mp4, boxes.json, meta.json, boxes_preview.mp4 (boxes drawn; check it by eye).
If several people are in view and the automatic target (largest*most central track) is wrong, pass
--point X,Y,FRAME (a pixel on the right person in that frame) to pick the track passing closest to it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from hmr_zoo import boxes as B  # noqa: E402
from hmr_zoo import common as C  # noqa: E402

COCO_PERSON = 1  # RF-DETR keeps original COCO category ids (person = 1)
FINEFS_OUT = C.FINEFS_ROOT / "outputs/stage2"


def cut_video(src: Path, start: float, dur: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
           "-an", "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=7.0)
    ap.add_argument("--finefs", help="finefs Stage 2 clip key, e.g. 3A_3A_0133")
    ap.add_argument("--sport", required=True)
    ap.add_argument("--name")
    ap.add_argument("--out", default=str(ROOT / "outputs/hmr_zoo"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0)
    ap.add_argument("--point", help="X,Y,FRAME pick the track nearest this pixel at that frame")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    if a.finefs:
        src_dir = FINEFS_OUT / a.finefs
        name = a.name or a.finefs
        out = Path(a.out) / a.sport / name
        out.mkdir(parents=True, exist_ok=True)
        sel = {r["jump"] + "_" + r["clip_id"]: r for r in __import__("csv").DictReader(open(C.FINEFS_ROOT / "notes/stage2_selected.csv"))}[a.finefs]
        mp4_src = Path("/home/allen/datasets/FineFS_5s/0_clips") / sel["jump"] / sel["clip_id"] / f"{sel['clip_id']}.mp4"
        shutil.copy2(mp4_src, out / "clip.mp4")
        shutil.copy2(src_dir / "boxes.json", out / "boxes.json")
        source = {"type": "finefs_stage2", "key": a.finefs, "video": str(mp4_src), "boxes": str(src_dir / "boxes.json")}
    else:
        src = Path(a.video)
        name = a.name or f"{src.stem}_{a.start:.1f}s"
        out = Path(a.out) / a.sport / name
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "clip.mp4").exists():
            cut_video(src, a.start, a.dur, out / "clip.mp4")
        from preprocessors.track_person import track_video

        tp = None
        if a.point:
            x, y, f = a.point.split(",")
            tp = (float(x), float(y), int(f))
        track_video(out / "clip.mp4", out, detector="medium", tracker="botsort-reid",
                    target="point" if tp else "auto", point=tp, device=a.device, gpu_mem_gb=a.gpu_mem_gb, preview=False)
        source = {"type": "video", "video": str(src), "start_sec": a.start, "dur_sec": a.dur,
                  "tracker": "RF-DETR-Medium + BoT-SORT + osnet_x1_0_sportsmot Re-ID",
                  "target": f"--point {a.point}" if a.point else "auto (area*centrality)"}

    W, H, fps, n = C.read_video_info(out / "clip.mp4")
    meta = {"sport": a.sport, "name": name, "width": W, "height": H, "fps": fps, "n_frames": n, "source": source,
            "note": a.note}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    from preprocessors.track_person import write_preview

    write_preview(out / "clip.mp4", json.loads((out / "boxes.json").read_text()), out / "boxes_preview.mp4", "padded_112")
    print(f"clip dir: {out}  ({n} frames, {W}x{H}, {fps:.2f} fps)")


if __name__ == "__main__":
    main()
