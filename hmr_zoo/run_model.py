"""Run ONE model on one or more clip directories (invoked by compare.py inside the model's conda env).

    <env python> -m hmr_zoo.run_model --model gvhmr --clips CLIP_DIR[,CLIP_DIR...] [--no-warmup]

For every clip the model's prediction is written to <clip_dir>/<model>.npz. The model is loaded once; unless
--no-warmup is given, the first clip is run once and discarded so fps/VRAM numbers exclude one-off CUDA/cuDNN init
(this is also what finefs_preproc did for the OnlineHMR timing).
"""

from __future__ import annotations

import sys

# Keep ~/.local user-site packages (e.g. a numpy 2.x / pytorch3d built for another python) behind the conda env's own
# site-packages. Must run before numpy is imported (finefs_preproc scripts did the same).
sys.path = [p for p in sys.path if ".local" not in p] + [p for p in sys.path if ".local" in p]

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmr_zoo import common as C  # noqa: E402
from hmr_zoo import models as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(M.SPECS))
    ap.add_argument("--clips", required=True, help="comma-separated clip directories")
    ap.add_argument("--box-key", default="padded_112")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--device", default="cuda:0", help="index as seen by this process (CUDA_VISIBLE_DEVICES applies)")
    args = ap.parse_args()

    clips = [C.Clip(p) for p in args.clips.split(",") if p]
    # resolve everything to absolute paths first: some models chdir into their own repo
    items = [(c, str(c.mp4.resolve()), str(c.boxes_txt(args.box_key).resolve()), c.npz(args.model).resolve()) for c in clips]
    det = M.build(args.model, args.device)

    if not args.no_warmup:
        print(f"[{args.model}] warmup pass on {items[0][0].key} (discarded)", flush=True)
        det.detect_video(items[0][1], items[0][2])

    for clip, mp4, bxt, out in items:
        t0 = time.perf_counter()
        r = det.detect_video(mp4, bxt)
        r["extra_meta"] = {**r.get("extra_meta", {}), "box_key": args.box_key, "clip": clip.key}
        C.save_prediction_npz(out, **r)
        T = r["joints3d_cam"].shape[0]
        print(f"[{args.model}] {clip.key} T={T} infer={r['infer_sec']:.2f}s ({T / r['infer_sec']:.1f} fps) "
              f"peak_vram={r['peak_vram_gb']:.2f}GB wall={time.perf_counter() - t0:.1f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
