"""Read gvhmr_fast per-video .npz files (numpy + scipy only, no torch) and turn them into other formats.

    from hmr_zoo.fast_io import load_fast, to_mstcn_features
    d = load_fast("outputs/finefs_full_test/npz/1001.npz")        # dict of arrays + d["meta"]
    runs = to_mstcn_features(d, zero_pelvis=True)                  # list of dict(start, end, start25, feat (72,T25), joints)

World coordinates are what is stored; zeroing the pelvis, re-sampling to 25 fps and shifting the height are done here
on demand (Allen 2026-11-23: "store global positions, normalise when used").

Henry's MS-TCN2 input format (see /home/allen/skating_jump_clips/scripts/features.py): 25 fps, Y up, metres,
pelvis X/Z = 0 in every frame (pelvis height kept), lowest-foot median height 0.366 m, body heading not normalised
(camera view). gvhmr_fast has its own world yaw, so `camera_aligned=True` (default) rotates each segment about the
vertical axis so the camera looks along +Z at its median pose, which is the closest to HybrIK's camera-view heading.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation as R

TARGET_LOWFOOT_Y = 0.366
FOOT_IDS = [7, 8, 10, 11]   # SMPL24: ankles and toes


def load_fast(path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d.pop("meta_json")))
    return d


def segments(d) -> list[tuple[int, int, bool]]:
    """(start, end_exclusive, vo_failed) of every segment (each has its own world frame)."""
    return [(int(a), int(b), bool(v)) for a, b, v in d["segments"]]


def camera_yaw(d, a: int, b: int) -> float:
    """Heading (rad about world +Y) of the camera's optical axis over frames [a,b), from the model's own outputs:
    R_c2w = R_world(global_orient) @ R_cam(global_orient)^T. Returns the circular mean over the segment."""
    Rw = R.from_rotvec(d["smpl_world_global_orient"][a:b].astype(np.float64)).as_matrix()
    Rc = R.from_rotvec(d["smpl_cam_global_orient"][a:b].astype(np.float64)).as_matrix()
    fwd = np.einsum("tij,j->ti", Rw @ np.transpose(Rc, (0, 2, 1)), np.array([0.0, 0.0, 1.0]))
    return float(np.arctan2(np.sin(np.arctan2(fwd[:, 0], fwd[:, 2])).mean(), np.cos(np.arctan2(fwd[:, 0], fwd[:, 2])).mean()))


def resample(j: np.ndarray, fps_in: float, fps_out: float = 25.0) -> np.ndarray:
    if abs(fps_in - fps_out) < 1e-3:
        return j
    T = len(j)
    t_in = np.arange(T) / fps_in
    t_out = np.arange(0, t_in[-1] + 1e-9, 1.0 / fps_out)
    flat = j.reshape(T, -1)
    return np.stack([np.interp(t_out, t_in, flat[:, k]) for k in range(flat.shape[1])], 1).reshape(len(t_out), *j.shape[1:])


def to_mstcn_features(d, zero_pelvis: bool = True, camera_aligned: bool = True, min_len: int = 0) -> list[dict]:
    """One entry per segment: {'start','end' (source frames), 'start25' (25 fps frame no.), 'feat' (72, T25), 'joints' (T25,24,3)}.

    zero_pelvis=True gives Henry's format (pelvis X/Z = 0, height kept, lowest-foot median -> 0.366 m).
    zero_pelvis=False keeps the global horizontal position (world coordinates, only re-sampled and height-shifted)."""
    fps = float(d["fps"])
    out = []
    for a, b, _ in segments(d):
        if b - a < min_len:
            continue
        j = d["joints24_world"][a:b].astype(np.float64)
        if camera_aligned:
            yaw = camera_yaw(d, a, b)
            c, s = np.cos(-yaw), np.sin(-yaw)
            Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            j = j @ Ry.T
        j = resample(j, fps, 25.0)
        if zero_pelvis:
            j[:, :, 0] -= j[:, :1, 0]
            j[:, :, 2] -= j[:, :1, 2]
        low = np.min(j[:, FOOT_IDS, 1], axis=1)
        j[:, :, 1] += TARGET_LOWFOOT_Y - np.median(low)
        out.append({"start": a, "end": b, "start25": int(round(a * 25.0 / fps)),
                    "feat": j.reshape(len(j), 72).T.astype(np.float64), "joints": j})
    return out
