"""OnlineHMR (CVPR 2026, arXiv 2603.17355) camera-coordinate HMR core, run frame by frame (causal / online).

Official code (no LICENSE file upstream -> research use only): https://github.com/Tsukasane/Video-OnlineHMR,
checked out at finefs_preproc/external/OnlineHMR (override with ONLINEHMR_ROOT). Conda env `onlinehmr`;
needs LD_LIBRARY_PATH=/home/allen/miniconda3/envs/onlinehmr/lib.

Only the camera-coordinate network (HMR_VIMO: ViT-H backbone + causal KV-cache SMPL decoder, 2-frame feature cache,
official checkpoint results/onlinehmr/checkpoint.pth.tar) is run, on the shared boxes. The world-grounding part of the
official pipeline (MASt3R-SLAM + MoGe + SAM + ViTDet) is NOT run. Focal length = 0.8*max(W,H), the official
run_custom.py default when no calibration is given.

    LD_LIBRARY_PATH=/home/allen/miniconda3/envs/onlinehmr/lib \
    /home/allen/miniconda3/envs/onlinehmr/bin/python detectors/onlinehmr_detector.py \
        --input clip.mp4 --bbox-file boxes_padded_112.txt --output onlinehmr.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmr_zoo import common as C  # noqa: E402
from hmr_zoo.vram import ProcessVramMonitor  # noqa: E402

ONLINEHMR_ROOT = Path(os.environ.get("ONLINEHMR_ROOT", str(C.FINEFS_ROOT / "external/OnlineHMR")))


def _install_stub_modules() -> None:
    """The HMR core never calls ViTDet/SAM, but lib/pipeline/tools.py imports them at module level."""

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            raise RuntimeError("stubbed module used (detector side is not part of hmr_zoo)")

    def make(name: str) -> types.ModuleType:
        m = types.ModuleType(name)

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _Dummy

        m.__getattr__ = _getattr  # type: ignore[assignment]
        m.__path__ = []  # type: ignore[attr-defined]
        return m

    for name in ["detectron2", "detectron2.config", "detectron2.data", "detectron2.data.transforms",
                 "detectron2.checkpoint", "segment_anything", "pycocotools", "pycocotools.mask"]:
        if name not in sys.modules:
            mod = make(name)
            sys.modules[name] = mod
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, mod)


class OnlineHMRDetector:
    def __init__(self, device: str = "cuda:0", dilate: float = 1.2) -> None:
        _install_stub_modules()
        os.chdir(str(ONLINEHMR_ROOT))  # lib/models/smpl.py uses relative data/smpl paths
        sys.path.insert(0, str(ONLINEHMR_ROOT))
        import torch
        from lib.models import get_hmr_vimo
        from lib.models.smpl import SMPL

        self.torch, self.dev, self.dilate = torch, "cuda", dilate
        t0 = time.perf_counter()
        self.model = get_hmr_vimo(checkpoint="./results/onlinehmr/checkpoint.pth.tar")
        self.smpl_layer = SMPL(gender="neutral").to("cuda")
        self.j17, self.faces = C.load_smpl_j17_and_faces()
        print(f"[OnlineHMR] loaded in {time.perf_counter() - t0:.1f}s; torch {torch.__version__}", flush=True)

    def detect_video(self, video_path: str, bbox_file: str) -> dict:
        torch, dev = self.torch, self.dev
        from lib.core import constants
        from lib.utils.imutils import boxes_2_cs, crop
        from torchvision.transforms import Compose, Normalize, ToTensor

        norm = Compose([ToTensor(), Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)])
        frames = C.read_frames(Path(video_path))
        H, W = frames[0].shape[:2]
        boxes = C.load_boxes_txt(bbox_file)
        T = min(len(frames), len(boxes))
        frames, boxes = frames[:T], boxes[:T]
        focal = max(W, H) * 0.8
        centers, scales = boxes_2_cs(boxes)
        focal_t = torch.tensor([focal], dtype=torch.float32, device=dev)
        ictr_t = torch.tensor(np.array([[W / 2.0, H / 2.0]], dtype=np.float32), device=dev)

        cache, verts = None, []
        pre_ms, net_ms, smpl_ms = [], [], []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with ProcessVramMonitor() as mon:
            wall0 = time.perf_counter()
            with torch.no_grad():
                for t in range(T):
                    t0 = time.perf_counter()
                    img = frames[t][:, :, ::-1]
                    sc = float(scales[t]) * self.dilate
                    crop_img = crop(img, centers[t], sc, [256, 256], rot=0).astype("uint8")
                    batch = {"img": norm(crop_img).unsqueeze(0).to(dev),
                             "scale": torch.tensor([sc], dtype=torch.float32, device=dev),
                             "center": torch.tensor(centers[t][None], dtype=torch.float32, device=dev),
                             "img_focal": focal_t, "img_center": ictr_t}
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    out, _, cache = self.model.inference_forward(batch, cache=cache)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    rot = out["pred_rotmat"]
                    sm = self.smpl_layer(body_pose=rot[:, 1:], global_orient=rot[:, [0]], betas=out["pred_shape"],
                                         transl=out["trans_full"].squeeze(1), pose2rot=False, default_smpl=True)
                    verts.append(sm.vertices[0].float().cpu().numpy())
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    pre_ms.append((t1 - t0) * 1e3)
                    net_ms.append((t2 - t1) * 1e3)
                    smpl_ms.append((t3 - t2) * 1e3)
            wall = time.perf_counter() - wall0
        tot = np.array(pre_ms) + np.array(net_ms) + np.array(smpl_ms)
        peak = max(torch.cuda.max_memory_allocated() / 1024**3, mon.peak_gb)
        verts = np.stack(verts, 0).astype(np.float32)
        j3d = np.einsum("jv,tvd->tjd", self.j17, verts).astype(np.float32)
        K = C.make_intrinsics(focal, focal, W / 2.0, H / 2.0)
        return dict(mesh_type="smpl", verts_cam=verts, joints3d_cam=j3d, K_fullimg=K, faces=self.faces,
                    infer_sec=wall, peak_vram_gb=peak,
                    extra_meta={"model": "onlinehmr (HMR_VIMO camera-coord, official checkpoint, frame-by-frame online)",
                                "focal_px": float(focal), "focal_rule": "0.8*max(W,H)",
                                "lat_ms_p50": float(np.percentile(tot, 50)), "lat_ms_p95": float(np.percentile(tot, 95)),
                                "internal_crop": "1.20x square (256x256)"})


def main() -> None:
    ap = argparse.ArgumentParser(description="OnlineHMR detector (camera-coord SMPL, shared boxes)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--bbox-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    inp, bbx, out = os.path.abspath(a.input), os.path.abspath(a.bbox_file), os.path.abspath(a.output)
    r = OnlineHMRDetector(a.device).detect_video(inp, bbx)
    C.save_prediction_npz(Path(out), **r)


if __name__ == "__main__":
    main()
