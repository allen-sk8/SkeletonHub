"""NLF (Neural Localizer Fields, nlf_l_multi_0.3.2 TorchScript) detector: camera-frame SMPL vertices.

Runs in conda env `finefs_pose`. Weights: finefs_preproc/data/weights/nlf/nlf_l_multi_0.3.2.torchscript
(override with NLF_WEIGHTS). Intrinsics are not estimated: a fixed 55-degree FOV pinhole is assumed
(same as finefs_preproc/scripts/stage2_infer_nlf.py).

    /home/allen/miniconda3/envs/finefs_pose/bin/python detectors/nlf_detector.py \
        --input clip.mp4 --bbox-file boxes_padded_112.txt --output nlf.npz

License: NLF code MIT; weights/SMPL are research-only (non-commercial).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmr_zoo import common as C  # noqa: E402

NLF_WEIGHTS = Path(os.environ.get("NLF_WEIGHTS", str(C.FINEFS_ROOT / "data/weights/nlf/nlf_l_multi_0.3.2.torchscript")))


class NLFDetector:
    def __init__(self, device: str = "cuda:0", batch_size: int = 16, num_aug: int = 1) -> None:
        import torch
        import torchvision  # noqa: F401 - registers torchvision::nms before torch.jit.load

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size, self.num_aug = batch_size, num_aug
        t0 = time.perf_counter()
        self.model = torch.jit.load(str(NLF_WEIGHTS), map_location=self.device)
        self.model.eval()
        self.j17, self.faces = C.load_smpl_j17_and_faces()
        print(f"[NLF] loaded {NLF_WEIGHTS.name} in {time.perf_counter() - t0:.2f}s", flush=True)

    def detect_video(self, video_path: str, bbox_file: str, fov_deg: float = 55.0) -> dict:
        torch, dev = self.torch, self.device
        frames = C.read_frames(Path(video_path), bgr=False)
        h, w = frames[0].shape[:2]
        boxes = C.load_boxes_txt(bbox_file)
        T = min(len(frames), len(boxes))
        frames, boxes = frames[:T], boxes[:T]
        focal = max(w, h) / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        K = C.make_intrinsics(focal, focal, w / 2.0, h / 2.0)
        K_t = torch.from_numpy(K).to(device=dev, dtype=torch.float32).unsqueeze(0)

        verts = []
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, T, self.batch_size):
                e = min(T, s + self.batch_size)
                imgs = torch.from_numpy(np.stack(frames[s:e], 0)).permute(0, 3, 1, 2).contiguous().to(dev)
                box_list = [torch.tensor([[float(x1), float(y1), max(float(x2 - x1), 2.0), max(float(y2 - y1), 2.0), 1.0]],
                                         dtype=torch.float32, device=dev) for x1, y1, x2, y2 in boxes[s:e]]
                out = self.model.estimate_smpl_batched(imgs, box_list, intrinsic_matrix=K_t, num_aug=self.num_aug,
                                                       internal_batch_size=max(self.batch_size, 32), model_name="smpl")
                for i in range(e - s):
                    v = out["vertices3d"][i]
                    # NLF returns SMPL vertices in millimetres.
                    verts.append(np.zeros((6890, 3), np.float32) if v.shape[0] == 0
                                 else v[0].detach().cpu().numpy().astype(np.float32) / 1000.0)
        torch.cuda.synchronize(dev)
        infer_sec = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        verts = np.stack(verts, 0)
        j3d = np.einsum("jv,tvd->tjd", self.j17, verts).astype(np.float32)
        return dict(mesh_type="smpl", verts_cam=verts, joints3d_cam=j3d, K_fullimg=K, faces=self.faces,
                    infer_sec=infer_sec, peak_vram_gb=peak,
                    extra_meta={"model": "nlf_l_multi_0.3.2", "num_aug": self.num_aug, "batch_size": self.batch_size,
                                "fov_deg": fov_deg, "internal_crop": "1.00x (384x384)"})


def main() -> None:
    ap = argparse.ArgumentParser(description="NLF detector (camera-frame SMPL, shared boxes)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--bbox-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    r = NLFDetector(a.device).detect_video(os.path.abspath(a.input), os.path.abspath(a.bbox_file))
    C.save_prediction_npz(Path(os.path.abspath(a.output)), **r)


if __name__ == "__main__":
    main()
