"""WHAM (ViTPose + HMR2 ViT features + WHAM temporal decoder) detector, camera-frame SMPL output.

Runs WHAM in its own conda env (`wham`) on a video with externally supplied per-frame person boxes
(same shared boxes as every other model in hmr_zoo). DPVO is not installed, so WHAM runs in its local-camera
mode (identity camera angular velocity), exactly as finefs_preproc/scripts/stage2_infer_wham.py did.

    /home/allen/miniconda3/envs/wham/bin/python detectors/wham_detector.py \
        --input clip.mp4 --bbox-file boxes_padded_112.txt --output wham.npz

License: WHAM code MIT-style research license; SMPL body model is non-commercial (research only).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# chumpy (used by smplx for SMPL_NEUTRAL.pkl) imports legacy numpy aliases removed in NumPy >= 1.24.
for _attr, _val in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
                    ("object", object), ("unicode", str), ("str", str)]:
    if _attr not in np.__dict__:
        setattr(np, _attr, _val)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmr_zoo import common as C  # noqa: E402

WHAM_ROOT = Path(os.environ.get("WHAM_ROOT", str(C.SKELETONHUB_ROOT / "external/WHAM")))
WHAM_BROKEN_VIT = C.SKELETONHUB_ROOT / "external/WHAM_broken/third-party/ViTPose"


def xyxy_to_cxcys(boxes_xyxy: np.ndarray) -> np.ndarray:
    cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
    scale = np.maximum(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / 200.0
    return np.stack([cx, cy, scale], axis=-1).astype(np.float32)


class WHAMDetector:
    def __init__(self, device: str = "cuda:0") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        os.chdir(str(WHAM_ROOT))  # WHAM config / checkpoint paths are relative to its root
        sys.path.insert(0, str(WHAM_ROOT))
        from configs.config import get_cfg_defaults
        from lib.data.datasets import CustomDataset
        from lib.models import build_body_model, build_network
        from lib.models.preproc.detector import VIT_DIR
        from lib.models.preproc.extractor import FeatureExtractor
        from lib.utils.imutils import avg_preds
        from mmpose.apis import inference_top_down_pose_model, init_pose_model

        self.CustomDataset, self.avg_preds = CustomDataset, avg_preds
        self.inference_top_down_pose_model = inference_top_down_pose_model
        cfg = get_cfg_defaults()
        cfg.merge_from_file("configs/yamls/demo.yaml")
        self.cfg = cfg
        t_load = time.perf_counter()
        vit_dir = (str(WHAM_BROKEN_VIT)
                   if (WHAM_BROKEN_VIT / "configs").exists() and not (Path(VIT_DIR) / "configs").exists() else VIT_DIR)
        pose_cfg = os.path.join(vit_dir, "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py")
        pose_ckpt = os.path.join(str(WHAM_ROOT), "checkpoints", "vitpose-h-multi-coco.pth")
        self.pose_model = init_pose_model(pose_cfg, pose_ckpt, device=str(device))
        self.extractor = FeatureExtractor(str(device), cfg.FLIP_EVAL)
        smpl = build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
        self.network = build_network(cfg, smpl)
        self.network.eval()
        self.j17, self.faces = C.load_smpl_j17_and_faces()
        print(f"[WHAM] loaded in {time.perf_counter() - t_load:.2f}s", flush=True)

    def detect_video(self, video_path: str, bbox_file: str) -> dict:
        torch = self.torch
        cfg, network = self.cfg, self.network
        frames_bgr = C.read_frames(Path(video_path))
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        H, W = frames_bgr[0].shape[:2]
        boxes = C.load_boxes_txt(bbox_file)
        T = min(len(frames_bgr), len(boxes))
        frames_bgr, boxes = frames_bgr[:T], boxes[:T]

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        kpts = []
        for t in range(T):
            b = boxes[t]
            person = [{"bbox": np.array([b[0], b[1], b[2], b[3], 1.0], dtype=np.float32)}]
            res, _ = self.inference_top_down_pose_model(self.pose_model, frames_bgr[t], person_results=person,
                                                        format="xyxy", return_heatmap=False, outputs=None)
            kpts.append(np.asarray(res[0]["keypoints"], dtype=np.float32) if res else np.zeros((17, 3), np.float32))
        track = defaultdict(list)
        track["frame_id"] = np.arange(T, dtype=np.int64)
        track["bbox"] = xyxy_to_cxcys(boxes)
        track["keypoints"] = np.stack(kpts, 0)
        tracking = {0: track}

        with torch.no_grad():
            tracking = self.extractor.run(str(video_path), tracking)
            slam = np.zeros((T, 7), dtype=np.float32)
            slam[:, 6] = 1.0  # identity quaternion: local camera mode (no DPVO)
            dataset = self.CustomDataset(cfg, tracking, slam, W, H, fps)

            def fwd(flip):
                _, x, inits, feats, mask, init_root, cam_angvel, _, kw = dataset.load_data(0, flip)
                return network(x, inits, feats, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                               return_y_up=True, **kw), cam_angvel, kw

            if cfg.FLIP_EVAL:
                flipped, _, _ = fwd(True)
                pred, cam_angvel, kwargs = fwd(False)
                fpose = flipped["pose"].squeeze(0).reshape(-1, 24, 6)
                fshape = flipped["betas"].squeeze(0)
                pose = pred["pose"].squeeze(0).reshape(-1, 24, 6)
                shape = pred["betas"].squeeze(0)
                avg_pose, avg_shape = self.avg_preds(pose, shape, fpose, fshape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped["contact"][..., [2, 3, 0, 1]] + pred["contact"]) / 2.0
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            else:
                pred, _, _ = fwd(False)

        torch.cuda.synchronize(self.device)
        infer_sec = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
        verts = (pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)).detach().cpu().numpy().astype(np.float32)
        if verts.ndim == 4:
            verts = verts[0]
        assert verts.shape == (T, 6890, 3), verts.shape
        j3d = np.einsum("jv,tvd->tjd", self.j17, verts).astype(np.float32)
        K = dataset.intrinsics.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return dict(mesh_type="smpl", verts_cam=verts, joints3d_cam=j3d, K_fullimg=K, faces=self.faces,
                    infer_sec=infer_sec, peak_vram_gb=peak,
                    extra_meta={"model": f"{Path(cfg.TRAIN.CHECKPOINT).stem} (no DPVO)", "flip_eval": bool(cfg.FLIP_EVAL),
                                "internal_crop": "1.00x square (256x256 ViT feature, 256x192 ViTPose)"})


def main() -> None:
    ap = argparse.ArgumentParser(description="WHAM detector (camera-frame SMPL, shared boxes)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--bbox-file", required=True)
    ap.add_argument("--output", required=True, help="output .npz (hmr_zoo prediction format)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    det = WHAMDetector(a.device)
    r = det.detect_video(os.path.abspath(a.input), os.path.abspath(a.bbox_file))
    C.save_prediction_npz(Path(os.path.abspath(a.output)), **r)


if __name__ == "__main__":
    main()
