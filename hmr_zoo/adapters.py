"""Adapters that give SkeletonHub's pre-existing detectors the hmr_zoo interface.

Each adapter exposes ``detect_video(video_path, bbox_file) -> dict`` returning the keys of
``hmr_zoo.common.save_prediction_npz``. Code is ported from finefs_preproc/scripts/stage2_infer_skeletonhub.py
(HybrIK / GVHMR / InstantHMR, run 2026-11-15) so the numbers are comparable with that report.

These adapters import SkeletonHub's own detectors, which locate their weights relative to the SkeletonHub checkout
that holds the git-ignored external/ and model_files/ trees, so they run against SKELETONHUB_ROOT (main checkout,
read-only) rather than against this worktree.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C
from .vram import ProcessVramMonitor

# Prevent ~/.local py3.10 packages (e.g. an incompatible pytorch3d) from shadowing the conda env.
sys.path = [p for p in sys.path if ".local" not in p] + [p for p in sys.path if ".local" in p]


def _use_skeletonhub_root() -> None:
    root = str(C.SKELETONHUB_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


class HybrIKDetector:
    def __init__(self, device: str = "cuda:0") -> None:
        _use_skeletonhub_root()
        os.chdir(str(C.SKELETONHUB_ROOT))
        sys.path.insert(0, str(C.SKELETONHUB_ROOT / "external" / "Hybrik"))
        import torch
        from utils.hybrik.service_fast import HybrikServiceFast

        self.torch = torch
        self.j17, self.faces = C.load_smpl_j17_and_faces()
        self.baseline_mb = ProcessVramMonitor().baseline_mb
        t0 = time.perf_counter()
        self.service = HybrikServiceFast(gpu=0, skip_detection=True)
        print(f"[HybrIK] loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    def detect_video(self, video_path: str, bbox_file: str) -> dict:
        import tempfile

        from utils.hybrik.service_fast import HybrikFastRequest

        torch = self.torch
        w, h, _, _ = C.read_video_info(Path(video_path))
        raw_dir = Path(tempfile.mkdtemp(prefix="hybrik_raw_", dir=str(Path(video_path).parent)))
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        with ProcessVramMonitor(baseline_mb=self.baseline_mb) as mon:
            self.service.process_video(HybrikFastRequest(
                video_path=video_path, output_dir=str(raw_dir), save_name="skeleton", gpu=0, flip_test=True,
                skip_detection=True, batch_size=32, save_vertices=True, bbox_file=bbox_file))
        torch.cuda.synchronize(0)
        infer_sec = time.perf_counter() - t0
        torch_peak = torch.cuda.max_memory_allocated(0) / 1024**3
        with (raw_dir / "smpl.pk").open("rb") as f:
            d = pickle.load(f)
        verts = np.asarray(d["pred_vertices"], np.float32) + np.asarray(d["transl"], np.float32)[:, None, :]
        bbox_used = np.asarray(d["bbox"], np.float32)
        T = verts.shape[0]
        j3d = np.einsum("jv,tvd->tjd", self.j17, verts).astype(np.float32)
        K = np.zeros((T, 3, 3), np.float32)
        for t in range(T):
            bw = max(float(bbox_used[t, 2] - bbox_used[t, 0]), 2.0)
            f = 1000.0 * bw / 256.0
            K[t] = C.make_intrinsics(f, f, w / 2.0, h / 2.0)
        return dict(mesh_type="smpl", verts_cam=verts, joints3d_cam=j3d, K_fullimg=K, faces=self.faces,
                    infer_sec=infer_sec, peak_vram_gb=max(torch_peak, mon.peak_gb),
                    extra_meta={"model": "hybrik_hrnet48", "flip_test": True, "torch_peak_gb": float(torch_peak),
                                "internal_crop": "1.25x square (256x256)", "raw_dir": str(raw_dir)})


class GVHMRDetector:
    def __init__(self, device: str = "cuda:0", static_cam: bool = False) -> None:
        _use_skeletonhub_root()
        os.chdir(str(C.SKELETONHUB_ROOT / "external" / "gvhmr"))
        import torch
        from detectors.gvhmr_detector import GVHMRDetector as _G

        self.torch, self.static_cam = torch, static_cam
        self.j17, self.faces = C.load_smpl_j17_and_faces()
        self.baseline_mb = ProcessVramMonitor().baseline_mb
        t0 = time.perf_counter()
        self.det = _G(device=device)
        print(f"[GVHMR] loaded in {time.perf_counter() - t0:.2f}s (static_cam={static_cam})", flush=True)

    def detect_video(self, video_path: str, bbox_file: str) -> dict:
        torch = self.torch
        kw = dict(rebase=False, run_smpl24=False, run_body25=False, run_h36m17=False, bbox_file=bbox_file)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        used_static = self.static_cam
        with ProcessVramMonitor(baseline_mb=self.baseline_mb) as mon:
            try:
                res = self.det.detect_video(video_path=video_path, static_cam=self.static_cam, **kw)
            except Exception as exc:  # SimpleVO can fail on tiny motion; same fallback as finefs
                if self.static_cam:
                    raise
                print(f"[GVHMR] SimpleVO failed ({exc}); falling back to static_cam=True", flush=True)
                used_static = True
                res = self.det.detect_video(video_path=video_path, static_cam=True, **kw)
        torch.cuda.synchronize(0)
        infer_sec = time.perf_counter() - t0
        torch_peak = torch.cuda.max_memory_allocated(0) / 1024**3
        verts = np.asarray(res["vertices_cam"], np.float32)
        j3d = np.einsum("jv,tvd->tjd", self.j17, verts).astype(np.float32)
        return dict(mesh_type="smpl", verts_cam=verts, joints3d_cam=j3d,
                    K_fullimg=np.asarray(res["K_fullimg"], np.float32), faces=self.faces,
                    infer_sec=infer_sec, peak_vram_gb=max(torch_peak, mon.peak_gb),
                    extra_meta={"model": "gvhmr", "static_cam": bool(used_static), "torch_peak_gb": float(torch_peak),
                                "internal_crop": "1.20x aspect 3:4 (256x192 ViTPose, 256x256 ViT)"})


class InstantHMRDetector:
    def __init__(self, device: str = "cuda:0") -> None:
        _use_skeletonhub_root()
        os.chdir(str(C.SKELETONHUB_ROOT))
        sys.path.insert(0, str(C.SKELETONHUB_ROOT / "external" / "InstantHMR"))
        import detectors.instanthmr_detector as mod  # triggers cuDNN 9 preload

        if not mod.HAS_INSTANTHMR:
            raise RuntimeError(f"InstantHMR unavailable: {mod.INSTANTHMR_ERROR}")
        from instanthmr import InstantHMR, resolve_model
        from instanthmr.smoothing import PoseFilter, TemporalSmoother

        self.PoseFilter, self.TemporalSmoother = PoseFilter, TemporalSmoother
        self.baseline_mb = ProcessVramMonitor().baseline_mb
        t0 = time.perf_counter()
        self.hmr = InstantHMR(resolve_model("medium", 2), device="cuda", max_batch=1, pose_filter=PoseFilter("one-euro"))
        self.faces = np.asarray(self.hmr.faces, dtype=np.int32)
        print(f"[InstantHMR] loaded medium-fit2 in {time.perf_counter() - t0:.2f}s", flush=True)

    def detect_video(self, video_path: str, bbox_file: str) -> dict:
        import cv2

        hmr = self.hmr
        hmr.pose_filter = self.PoseFilter("one-euro")  # reset causal temporal filters per clip
        smoother = self.TemporalSmoother(beta_scale=1.0)
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        frames = C.read_frames(Path(video_path), bgr=False)
        H, W = frames[0].shape[:2]
        boxes = C.load_boxes_txt(bbox_file)
        T = min(len(frames), len(boxes))
        frames, boxes = frames[:T], boxes[:T]
        verts, mhr70, last = [], [], None
        focal_default = 1.05 * math.hypot(H, W)
        f_val = focal_default
        t0 = time.perf_counter()
        with ProcessVramMonitor(baseline_mb=self.baseline_mb) as mon:
            for i in range(T):
                ts = i / fps
                preds = hmr.predict(frames[i], boxes[i:i + 1], timestamp=ts)
                if len(preds) > 0:
                    preds = smoother(preds, timestamp=ts)
                    last = preds[0]
                p = last
                if p is not None:
                    cam_t = np.asarray(p.cam_trans, np.float32)
                    verts.append(np.asarray(p.vertices_local, np.float32) + cam_t[None, :])
                    mhr70.append(np.asarray(p.joints_3d_local, np.float32) + cam_t[None, :])
                    f_val = float(getattr(p, "focal_length", focal_default) or focal_default)
                else:
                    verts.append(np.zeros((18439, 3), np.float32))
                    mhr70.append(np.zeros((70, 3), np.float32))
        infer_sec = time.perf_counter() - t0
        j3d = C.mhr70_to_h36m17(np.stack(mhr70, 0)).astype(np.float32)
        return dict(mesh_type="mhr", verts_cam=np.stack(verts, 0), joints3d_cam=j3d,
                    K_fullimg=C.make_intrinsics(f_val, f_val, W / 2.0, H / 2.0), faces=self.faces,
                    infer_sec=infer_sec, peak_vram_gb=mon.peak_gb,
                    extra_meta={"model": "instanthmr_medium_fit2", "pose_filter": "one-euro",
                                "temporal_smoother_beta_scale": 1.0,
                                "internal_crop": "2-crop multi-scale (0.9x1.2=1.08x, 0.7x1.2=0.84x)"})


class RTMPose3DDetector:
    """RTMW3D-x (mmpose project rtmpose3d): 133 whole-body keypoints, 2D in image pixels + metric-ish 3D.

    Skeleton only (no mesh). The 17 COCO body joints are mapped to H36M-17; joints2d are the model's own
    image-space predictions (``transformed_keypoints``), joints3d_cam are the camera-axis-aligned 3D keypoints
    (root-relative depth, so the depth-jump metric is not comparable with the mesh models).
    """

    def __init__(self, device: str = "cuda:0") -> None:
        _use_skeletonhub_root()
        os.chdir(str(C.SKELETONHUB_ROOT))
        rtm = str(C.SKELETONHUB_ROOT / "external" / "mmpose" / "projects" / "rtmpose3d")
        sys.path.insert(0, rtm)
        import torch
        from mmpose.apis import inference_topdown, init_model
        import importlib

        importlib.import_module("rtmpose3d")  # registers the rtmpose3d models/heads with mmpose

        self.torch, self.inference_topdown = torch, inference_topdown
        cfg = os.path.join(rtm, "configs", "rtmw3d-x_8xb32_cocktail14-384x288.py")
        ckpt = "https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth"
        t0 = time.perf_counter()
        self.model = init_model(cfg, ckpt, device=device.lower())
        self.baseline_mb = ProcessVramMonitor().baseline_mb
        print(f"[RTMPose3D] loaded rtmw3d-x in {time.perf_counter() - t0:.2f}s", flush=True)

    def detect_video(self, video_path: str, bbox_file: str) -> dict:
        torch = self.torch
        frames = C.read_frames(Path(video_path))
        H, W = frames[0].shape[:2]
        boxes = C.load_boxes_txt(bbox_file)
        T = min(len(frames), len(boxes))
        k3, k2 = [], []
        last3, last2 = np.zeros((133, 3), np.float32), np.zeros((133, 2), np.float32)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        with ProcessVramMonitor(baseline_mb=self.baseline_mb) as mon:
            for i in range(T):
                res = self.inference_topdown(self.model, frames[i], boxes[i:i + 1])
                if not res:
                    k3.append(last3.copy())
                    k2.append(last2.copy())
                    continue
                pi = res[0].pred_instances
                kp = np.asarray(pi.keypoints, np.float32).reshape(-1, 133, 3)[0]
                tk = np.asarray(pi.transformed_keypoints, np.float32).reshape(-1, 133, 2)[0]
                last3, last2 = kp, tk
                k3.append(kp)
                k2.append(tk)
        torch.cuda.synchronize(0)
        infer_sec = time.perf_counter() - t0
        k3, k2 = np.stack(k3, 0), np.stack(k2, 0)
        j3d = C.coco17_to_h36m17(k3[:, :17]).astype(np.float32)
        j2d = C.coco17_to_h36m17(k2[:, :17]).astype(np.float32)
        K = C.make_intrinsics(1145.0, 1145.0, W / 2.0, H / 2.0)
        return dict(mesh_type="skeleton", verts_cam=np.zeros((T, 1, 3), np.float32), joints3d_cam=j3d,
                    K_fullimg=K, faces=np.zeros((0, 3), np.int32), infer_sec=infer_sec,
                    peak_vram_gb=max(torch.cuda.max_memory_allocated(0) / 1024**3, mon.peak_gb), joints2d=j2d,
                    extra_meta={"model": "rtmw3d-x_cocktail14-384x288", "note": "skeleton only; 3D depth is root-relative"})


REGISTRY = {
    "hybrik": HybrIKDetector,
    "gvhmr": GVHMRDetector,
    "instanthmr": InstantHMRDetector,
    "rtmpose3d": RTMPose3DDetector,
}
