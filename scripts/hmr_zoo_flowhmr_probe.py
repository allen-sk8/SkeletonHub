"""Probe: does FlowHMR still run? (video -> SMPL-H motion), writing ONLY under outputs/hmr_zoo/flowhmr/.

FlowHMR (ReasonMotion_Predictor/baselines/flowhmr, read-only here) is a whole pipeline of its own: YOLOX detection and
tracking (it does NOT take the shared boxes), VGGT-Omega camera, SAM-3D-Body features, then a flow-matching HMR head;
it transcodes to 30 fps and outputs world-frame SMPL-H, not camera-frame meshes. So it is not part of the side-by-side
2D comparison; this script only checks that it still runs and records time and GPU memory.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
    /home/allen/miniconda3/envs/flowhmr/bin/python scripts/hmr_zoo_flowhmr_probe.py --video clip.mp4 --vggt-interval 10

Default (vggt-interval 1) peaks at ~30 GB; 10 should fit a 24 GB card. A hard memory cap (--max-gb, default 23.5) makes
it fail with OOM instead of growing past that.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOWHMR_ROOT = Path(os.environ.get("FLOWHMR_ROOT", "/home/allen/ReasonMotion_Predictor/baselines/flowhmr"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-root", default=str(ROOT / "outputs/hmr_zoo/flowhmr"))
    ap.add_argument("--vggt-interval", type=int, default=10)
    ap.add_argument("--max-gb", type=float, default=23.5)
    a = ap.parse_args()
    video = os.path.abspath(a.video)
    out_root = os.path.abspath(a.out_root)
    os.makedirs(out_root, exist_ok=True)

    import glob

    # system ffmpeg 4.4 lacks `-fps_mode`; the flowhmr env ships imageio_ffmpeg 7.x (same trick as detectors/flowhmr_detector.py)
    cands = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                                          "lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg*")))
    if cands:
        os.environ["FFMPEG"] = cands[-1]

    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gb * 2**30 / total), 0)
    sys.path.insert(0, str(FLOWHMR_ROOT))
    os.chdir(FLOWHMR_ROOT)  # asset paths are repo-relative
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    ckpt = FLOWHMR_ROOT / "checkpoints/flowhmr_latest/flowhmr_latest.ckpt"
    cfg = ckpt.parent / "config.yml"
    t0 = time.time()
    engine = V2MInferenceEngine.build(ckpt=str(ckpt), model_cfg_path=str(cfg), device="cuda", validation_steps=None,
                                      postprocess=False)
    print(f"[flowhmr] built in {time.time() - t0:.1f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    sub = engine.process_video(video, out_root, seed=0, cfg_scale=1.0, vggt_frame_interval=a.vggt_interval)
    sec = time.time() - t1
    print(f"[flowhmr] process_video {sec:.1f}s torch peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
          f"(reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB) -> {sub}", flush=True)


if __name__ == "__main__":
    main()
