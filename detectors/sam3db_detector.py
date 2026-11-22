"""SAM 3D Body (ViT-H, Meta) detector: MHR mesh + 70 keypoints in camera frame, per-frame (no temporal model).

Runs in conda env `finefs_pose`. Code/weights: finefs_preproc/external/sam-3d-body and
finefs_preproc/data/weights/sam-3d-body-vith (override with SAM3DB_ROOT / SAM3DB_WEIGHTS).
Output mesh is MHR (not SMPL); the 17 H36M joints used for metrics/overlay are derived from the MHR-70 keypoints
(hmr_zoo.common.mhr70_to_h36m17), and the full MHR mesh is projected for the overlay.

    /home/allen/miniconda3/envs/finefs_pose/bin/python detectors/sam3db_detector.py \
        --input clip.mp4 --bbox-file boxes_padded_112.txt --output sam3db.npz

License: SAM License (Meta); MHR body model separate license. Research use.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmr_zoo import common as C  # noqa: E402

SAM3DB_ROOT = Path(os.environ.get("SAM3DB_ROOT", str(C.FINEFS_ROOT / "external/sam-3d-body")))
SAM3DB_WEIGHTS = Path(os.environ.get("SAM3DB_WEIGHTS", str(C.FINEFS_ROOT / "data/weights/sam-3d-body-vith")))


class SAM3DBodyDetector:
    def __init__(self, device: str = "cuda:0", inference_type: str = "full") -> None:
        import torch

        sys.path.insert(0, str(SAM3DB_ROOT))
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        self.torch, self.device, self.inference_type = torch, torch.device(device), inference_type
        t0 = time.perf_counter()
        model, cfg = load_sam_3d_body(checkpoint_path=str(SAM3DB_WEIGHTS / "model.ckpt"),
                                      mhr_path=str(SAM3DB_WEIGHTS / "assets/mhr_model.pt"), device=str(self.device))
        self.estimator = SAM3DBodyEstimator(model, cfg)
        print(f"[SAM3DB] loaded ViT-H in {time.perf_counter() - t0:.2f}s", flush=True)

    def detect_video(self, video_path: str, bbox_file: str) -> dict:
        torch, dev = self.torch, self.device
        frames = C.read_frames(Path(video_path), bgr=False)
        h, w = frames[0].shape[:2]
        boxes = C.load_boxes_txt(bbox_file)
        T = min(len(frames), len(boxes))
        frames, boxes = frames[:T], boxes[:T]
        faces = np.asarray(self.estimator.faces, dtype=np.int32)
        n_verts = int(faces.max()) + 1
        verts, mhr70, Ks = [], [], []

        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for i in range(T):
            with contextlib.redirect_stdout(io.StringIO()):
                outs = self.estimator.process_one_image(frames[i], bboxes=boxes[i:i + 1], inference_type=self.inference_type)
            if not outs:
                f_def = float((w**2 + h**2) ** 0.5)
                verts.append(np.zeros((n_verts, 3), np.float32))
                mhr70.append(np.zeros((70, 3), np.float32))
                Ks.append(C.make_intrinsics(f_def, f_def, w / 2.0, h / 2.0))
                continue
            o = outs[0]
            cam_t = np.asarray(o["pred_cam_t"], dtype=np.float32)
            verts.append(np.asarray(o["pred_vertices"], dtype=np.float32) + cam_t[None, :])
            mhr70.append(np.asarray(o["pred_keypoints_3d"], dtype=np.float32) + cam_t[None, :])
            f = float(o["focal_length"])
            Ks.append(C.make_intrinsics(f, f, w / 2.0, h / 2.0))
        torch.cuda.synchronize(dev)
        infer_sec = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        j3d = C.mhr70_to_h36m17(np.stack(mhr70, 0)).astype(np.float32)
        return dict(mesh_type="mhr", verts_cam=np.stack(verts, 0), joints3d_cam=j3d, K_fullimg=np.stack(Ks, 0), faces=faces,
                    infer_sec=infer_sec, peak_vram_gb=peak,
                    extra_meta={"model": "sam-3d-body-vith", "inference_type": self.inference_type,
                                "internal_crop": "1.25x (512x512)", "mhr_verts": int(n_verts)})


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM 3D Body detector (MHR, shared boxes)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--bbox-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--inference-type", default="full", choices=["full", "body"])
    a = ap.parse_args()
    r = SAM3DBodyDetector(a.device, a.inference_type).detect_video(os.path.abspath(a.input), os.path.abspath(a.bbox_file))
    C.save_prediction_npz(Path(os.path.abspath(a.output)), **r)


if __name__ == "__main__":
    main()
