"""Shared constants, geometry helpers and the clip-directory convention for hmr_zoo.

A *clip directory* is the unit every model runs on:

    <clip_dir>/clip.mp4          the video segment (5-10 s)
    <clip_dir>/boxes.json        shared person boxes (keys: tight_clean, padded_112, ...)
    <clip_dir>/meta.json         sport, name, source, fps, size, n_frames
    <clip_dir>/<model>.npz       per-model prediction (written by hmr_zoo.run_model)
    <clip_dir>/compare_2d.mp4    side-by-side 2D reprojection comparison (hmr_zoo.render)
    <clip_dir>/metrics.csv       per-model metrics (hmr_zoo.render)

The npz format and metric definitions follow finefs_preproc/scripts/stage2_common.py and stage2_eval_render.py
(Allen's FineFS Stage 2 comparison, 2026-11-15) so numbers stay comparable with that report.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

# Runtime assets (SMPL files, external model code, weights) are read from the SkeletonHub main checkout,
# which has the git-ignored external/ and common_models/ trees. Read-only: nothing here writes into it.
SKELETONHUB_ROOT = Path(os.environ.get("SKELETONHUB_ROOT", "/home/allen/SkeletonHub"))
# finefs_preproc holds the NLF / SAM 3D Body / OnlineHMR code and weights (read-only use).
FINEFS_ROOT = Path(os.environ.get("FINEFS_ROOT", "/home/allen/finefs_preproc"))
ZOO_ROOT = Path(__file__).resolve().parents[1]  # checkout root (= SkeletonHub)

SMPL_NEUTRAL_PKL = SKELETONHUB_ROOT / "common_models/body_models/smpl/SMPL_NEUTRAL.pkl"
J17_REGRESSOR_NPY = SKELETONHUB_ROOT / "common_models/body_models/smpl/J_regressor_h36m.npy"

# SPIN/WHAM J_regressor_h36m puts +X (anatomical Left) at rows 1..3 and -X (Right) at rows 4..6,
# whereas standard H36M-17 and mhr70_to_h36m17 put Right Leg at 1..3 and Left Leg at 4..6.
SMPL_J17_TO_H36M17_PERM = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

H36M17_NAMES = [
    "Pelvis", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Neck", "Nose", "HeadTop",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]
H36M17_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
H36M17_BONES = [(H36M17_PARENTS[i], i) for i in range(1, 17)]
H36M17_RIGHT_EDGES = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}
H36M17_LEFT_EDGES = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}


# ----------------------------------------------------------------------------------------------- clip directory
class Clip:
    """Thin wrapper around a clip directory."""

    def __init__(self, clip_dir: str | Path) -> None:
        self.dir = Path(clip_dir)
        self.mp4 = self.dir / "clip.mp4"
        self.boxes_json = self.dir / "boxes.json"
        self.meta_json = self.dir / "meta.json"
        self.meta = json.loads(self.meta_json.read_text()) if self.meta_json.exists() else {}
        self.key = self.dir.name
        self.sport = self.meta.get("sport", self.dir.parent.name)

    def boxes(self, box_key: str = "padded_112") -> np.ndarray:
        data = json.loads(self.boxes_json.read_text())
        return np.asarray(data[box_key], dtype=np.float32)

    def boxes_txt(self, box_key: str = "padded_112") -> Path:
        """Write boxes as 'x1 y1 x2 y2' lines (the format SkeletonHub detectors take via --bbox-file)."""
        out = self.dir / f"boxes_{box_key}.txt"
        out.write_text("".join(f"{b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n" for b in self.boxes(box_key)))
        return out

    def npz(self, model: str) -> Path:
        return self.dir / f"{model}.npz"


# ----------------------------------------------------------------------------------------------- SMPL helpers
class _ChumpyStub:
    """Minimal stub to unpickle SMPL_NEUTRAL.pkl without installing legacy chumpy."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _SMPLUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def load_smpl_j17_and_faces() -> tuple[np.ndarray, np.ndarray]:
    """Return (J17_h36m_aligned (17, 6890), smpl_faces (13776, 3))."""
    j17 = np.load(J17_REGRESSOR_NPY).astype(np.float32)[SMPL_J17_TO_H36M17_PERM]
    with SMPL_NEUTRAL_PKL.open("rb") as f:
        smpl = _SMPLUnpickler(f, encoding="latin1").load()
    return j17, np.asarray(smpl["f"], dtype=np.int32)


def mhr70_to_h36m17(mhr_joints: np.ndarray) -> np.ndarray:
    """Convert MHR-70 joints (..., 70, D) to canonical anatomical H36M-17 (..., 17, D)."""
    shape = list(mhr_joints.shape)
    shape[-2] = 17
    h17 = np.zeros(shape, dtype=mhr_joints.dtype)
    pelvis = 0.5 * (mhr_joints[..., 9, :] + mhr_joints[..., 10, :])
    neck = mhr_joints[..., 69, :]
    h17[..., 0, :] = pelvis
    h17[..., 1, :] = mhr_joints[..., 10, :]   # RHip
    h17[..., 2, :] = mhr_joints[..., 12, :]   # RKnee
    h17[..., 3, :] = mhr_joints[..., 14, :]   # RAnkle
    h17[..., 4, :] = mhr_joints[..., 9, :]    # LHip
    h17[..., 5, :] = mhr_joints[..., 11, :]   # LKnee
    h17[..., 6, :] = mhr_joints[..., 13, :]   # LAnkle
    h17[..., 7, :] = 0.5 * (pelvis + neck)    # Spine
    h17[..., 8, :] = neck                     # Neck
    h17[..., 9, :] = mhr_joints[..., 0, :]    # Nose
    h17[..., 10, :] = mhr_joints[..., 0, :] + 0.4 * (mhr_joints[..., 0, :] - neck)
    h17[..., 11, :] = mhr_joints[..., 5, :]   # LShoulder
    h17[..., 12, :] = mhr_joints[..., 7, :]   # LElbow
    h17[..., 13, :] = mhr_joints[..., 62, :]  # LWrist
    h17[..., 14, :] = mhr_joints[..., 6, :]   # RShoulder
    h17[..., 15, :] = mhr_joints[..., 8, :]   # RElbow
    h17[..., 16, :] = mhr_joints[..., 41, :]  # RWrist
    return h17


def coco17_to_h36m17(coco: np.ndarray) -> np.ndarray:
    """COCO-17 body joints (..., 17, D) -> H36M-17 (..., 17, D).

    COCO order: 0 nose, 1 Leye, 2 Reye, 3 Lear, 4 Rear, 5 Lsho, 6 Rsho, 7 Lelb, 8 Relb, 9 Lwri, 10 Rwri,
    11 Lhip, 12 Rhip, 13 Lknee, 14 Rknee, 15 Lank, 16 Rank. Pelvis/Spine/Neck/HeadTop are derived (approximate),
    as is customary when lifting COCO to H36M.
    """
    shape = list(coco.shape)
    h = np.zeros(shape, dtype=coco.dtype)
    pelvis = 0.5 * (coco[..., 11, :] + coco[..., 12, :])
    neck = 0.5 * (coco[..., 5, :] + coco[..., 6, :])
    h[..., 0, :] = pelvis
    h[..., 1, :] = coco[..., 12, :]
    h[..., 2, :] = coco[..., 14, :]
    h[..., 3, :] = coco[..., 16, :]
    h[..., 4, :] = coco[..., 11, :]
    h[..., 5, :] = coco[..., 13, :]
    h[..., 6, :] = coco[..., 15, :]
    h[..., 7, :] = 0.5 * (pelvis + neck)
    h[..., 8, :] = neck
    h[..., 9, :] = coco[..., 0, :]
    h[..., 10, :] = coco[..., 0, :] + 0.4 * (coco[..., 0, :] - neck)
    h[..., 11, :] = coco[..., 5, :]
    h[..., 12, :] = coco[..., 7, :]
    h[..., 13, :] = coco[..., 9, :]
    h[..., 14, :] = coco[..., 6, :]
    h[..., 15, :] = coco[..., 8, :]
    h[..., 16, :] = coco[..., 10, :]
    return h


# ----------------------------------------------------------------------------------------------- projection / io
def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def project_points(pts3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (T, N, 3) OpenCV cam-space points via (3,3) or (T,3,3) intrinsics K -> (T, N, 2)."""
    K_use = np.broadcast_to(K[None], (pts3d.shape[0], 3, 3)) if K.ndim == 2 else K
    z = np.maximum(pts3d[..., 2:3], 0.05)
    xy = pts3d[..., :2] / z
    u = K_use[:, 0:1, 0] * xy[..., 0] + K_use[:, 0:1, 2]
    v = K_use[:, 1:2, 1] * xy[..., 1] + K_use[:, 1:2, 2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def save_prediction_npz(
    out_path: Path,
    mesh_type: str,
    verts_cam: np.ndarray,
    joints3d_cam: np.ndarray,
    K_fullimg: np.ndarray,
    faces: np.ndarray,
    infer_sec: float,
    peak_vram_gb: float,
    joints2d: np.ndarray | None = None,
    extra_meta: dict | None = None,
) -> None:
    """Save the canonical prediction .npz (same keys as finefs Stage 2).

    mesh_type is 'smpl', 'mhr' or 'skeleton' (no mesh: verts_cam is (T, 1, 3) zeros and faces is (0, 3)).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = int(joints3d_cam.shape[0])
    K_arr = np.asarray(K_fullimg, dtype=np.float32)
    if K_arr.ndim == 2:
        K_arr = np.repeat(K_arr[None], T, axis=0)
    j3d = np.asarray(joints3d_cam, dtype=np.float32)
    j2d = project_points(j3d, K_arr) if joints2d is None else np.asarray(joints2d, dtype=np.float32)
    np.savez(
        out_path,
        mesh_type=np.asarray(mesh_type),
        verts_cam=np.asarray(verts_cam, dtype=np.float32),
        joints3d_cam=j3d,
        joints2d=j2d,
        K_fullimg=K_arr,
        faces=np.asarray(faces, dtype=np.int32),
        infer_sec=np.asarray(infer_sec, dtype=np.float32),
        fps_infer=np.asarray(T / max(infer_sec, 1e-6), dtype=np.float32),
        peak_vram_gb=np.asarray(peak_vram_gb, dtype=np.float32),
        meta_json=np.asarray(json.dumps(extra_meta or {})),
    )


def read_video_info(mp4_path: Path) -> tuple[int, int, float, int]:
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n


def read_frames(mp4_path: Path, bgr: bool = True) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr if bgr else cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Zero frames in {mp4_path}")
    return frames


def load_boxes_txt(path: str | Path) -> np.ndarray:
    """Read 'x1 y1 x2 y2' per line (one per frame) -> (T, 4) float32."""
    return np.loadtxt(str(path), dtype=np.float32).reshape(-1, 4)
