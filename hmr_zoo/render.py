"""Metrics + side-by-side 2D reprojection comparison video (compare_2d.mp4) for one clip directory.

The drawing helpers (mesh painter, H36M-17 skeleton, crop window, ffmpeg writer) and the metric definitions are
taken from finefs_preproc/scripts/stage2_eval_render.py (Allen's FineFS Stage 2 comparison, 2026-11-15) so that
numbers and look match that report. Differences: any number of models, mesh-less (skeleton-only) models, and
a grid that grows to 3 rows when more than 8 cells are needed.

    python -m hmr_zoo.render --clip CLIP_DIR [--models hybrik,gvhmr,...]
Writes CLIP_DIR/compare_2d.mp4 and CLIP_DIR/metrics.csv.

Metric definitions (all camera space, per frame, T frames, fps = video fps):
  reproj_in_box_pct   % of the 17 joints whose 2D projection falls inside the shared 1.12x box
  center_err_pct_h    mean distance (% of tight-box height) between joint-bbox centre and tight-box centre
  proj_h_ratio        median(projected skeleton height / tight-box height)
  accel_jitter_mps2   mean ||second difference of root-relative joints|| * fps^2  (3D jitter; same units as finefs)
  bone_len_cv_pct     mean over 8 limb bones of std(length)/mean(length) * 100  (bone-length variation)
  pelvis_yaw_p95_deg  95th percentile of per-frame pelvis yaw change (LHip-RHip in the XZ plane)
  pelvis_abs_turn_deg sum of |per-frame yaw change|
  depth_z_jump_p95_m  95th percentile of per-frame |pelvis depth change|
  fps_infer / peak_vram_gb  measured by the runner (inference only, model load excluded)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmr_zoo import common as C  # noqa: E402
from hmr_zoo import models as M  # noqa: E402

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

LIMB_EDGES = [
    (1, 2), (2, 3),    # RThigh, RShin
    (4, 5), (5, 6),    # LThigh, LShin
    (11, 12), (12, 13),  # LUpperArm, LForearm
    (14, 15), (15, 16),  # RUpperArm, RForearm
]

# BGR colors for 2D skeleton edges: Blue=Left, Red=Right, Green=Spine/Head
COLOR_LEFT = (255, 130, 40)
COLOR_RIGHT = (45, 70, 255)
COLOR_CENTER = (65, 230, 95)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size, index=3)


def draw_zh_banner(w: int, h: int, text: str, font_sz: int = 18, bg=(26, 30, 38), fg=(240, 244, 250)) -> np.ndarray:
    pil = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(pil)
    font = load_font(font_sz)
    bbox = draw.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    draw.text((10, max(1, (h - th) // 2 - 2)), text, font=font, fill=fg)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def ensure_outward_faces(verts_seq: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Verify triangle winding produces outward normals on the first non-zero frame."""
    for v in verts_seq:
        if np.any(v != 0):
            v0 = v[faces[:, 0]]
            v1 = v[faces[:, 1]]
            v2 = v[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            centroids = (v0 + v1 + v2) / 3.0
            center = v.mean(axis=0)
            mean_dot = float(((centroids - center) * normals).sum(axis=-1).mean())
            if mean_dot < 0:
                return faces[:, [0, 2, 1]].copy()
            return faces
    return faces


def render_mesh_painter(
    canvas_bgr: np.ndarray,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    alpha: float = 1.0,
    base_bgr: tuple[int, int, int] = (225, 195, 165),
) -> np.ndarray:
    """Fast CPU painter's-algorithm shaded mesh renderer into canvas_bgr."""
    if not np.any(verts_cam != 0):
        return canvas_bgr
    H, W = canvas_bgr.shape[:2]
    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    tri_c = (v0 + v1 + v2) * (1.0 / 3.0)
    normals = np.cross(v1 - v0, v2 - v0)

    # Front-facing check in Y-down Z-fwd camera: ray +tri_c opposes outward normal.
    keep = ((normals * tri_c).sum(axis=-1) < 0.0) & (tri_c[:, 2] > 0.15)
    if not np.any(keep):
        return canvas_bgr

    faces_k = faces[keep]
    z_k = tri_c[keep, 2]
    n_k = normals[keep]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z_safe = np.maximum(verts_cam[:, 2:3], 0.1)
    uv = np.empty((verts_cam.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (verts_cam[:, 0:1] / z_safe[:, 0:1])[:, 0] * fx + cx
    uv[:, 1] = (verts_cam[:, 1:2] / z_safe[:, 0:1])[:, 0] * fy + cy

    pts = np.rint(uv[faces_k]).astype(np.int32)  # (M, 3, 2)
    xmin = pts[:, :, 0].min(axis=1)
    xmax = pts[:, :, 0].max(axis=1)
    ymin = pts[:, :, 1].min(axis=1)
    ymax = pts[:, :, 1].max(axis=1)
    in_bounds = (xmax >= 0) & (xmin < W) & (ymax >= 0) & (ymin < H)
    if not np.any(in_bounds):
        return canvas_bgr

    pts = pts[in_bounds]
    z_k = z_k[in_bounds]
    n_k = n_k[in_bounds]

    order = np.argsort(-z_k, kind="mergesort")
    pts_s = pts[order]
    n_s = n_k[order]

    n_norm = np.linalg.norm(n_s, axis=-1, keepdims=True) + 1e-8
    n_unit = n_s / n_norm
    light = np.array([-0.24, -0.45, -0.86], dtype=np.float32)
    diff = np.maximum(0.0, n_unit @ light)
    rim = np.maximum(0.0, 1.0 - np.abs(n_unit[:, 2]))
    shade = np.clip(0.30 + 0.64 * diff + 0.12 * rim, 0.22, 1.05)
    cols = np.clip(np.rint(shade[:, None] * np.asarray(base_bgr, dtype=np.float32)), 1, 255).astype(np.uint8)

    mesh_buf = np.zeros_like(canvas_bgr) if alpha < 0.999 else canvas_bgr
    for idx in range(len(pts_s)):
        c = cols[idx]
        cv2.fillConvexPoly(mesh_buf, pts_s[idx], (int(c[0]), int(c[1]), int(c[2])), lineType=cv2.LINE_8)

    if alpha < 0.999:
        mask = np.any(mesh_buf > 0, axis=-1)
        out = canvas_bgr.copy()
        out[mask] = (alpha * mesh_buf[mask].astype(np.float32) + (1.0 - alpha) * canvas_bgr[mask].astype(np.float32)).astype(np.uint8)
        return out
    return mesh_buf


def draw_h36m17_skeleton(img: np.ndarray, j2d: np.ndarray, thickness: int = 2) -> np.ndarray:
    H, W = img.shape[:2]
    if not np.any(j2d != 0):
        return img
    for a, b in C.H36M17_BONES:
        p1 = (int(round(float(j2d[a, 0]))), int(round(float(j2d[a, 1]))))
        p2 = (int(round(float(j2d[b, 0]))), int(round(float(j2d[b, 1]))))
        if max(abs(p1[0]), abs(p2[0])) > W * 2 or max(abs(p1[1]), abs(p2[1])) > H * 2:
            continue
        if (a, b) in C.H36M17_RIGHT_EDGES:
            col = COLOR_RIGHT
        elif (a, b) in C.H36M17_LEFT_EDGES:
            col = COLOR_LEFT
        else:
            col = COLOR_CENTER
        cv2.line(img, p1, p2, col, thickness, lineType=cv2.LINE_AA)
    for k in range(len(j2d)):
        p = (int(round(float(j2d[k, 0]))), int(round(float(j2d[k, 1]))))
        if -50 <= p[0] <= W + 50 and -50 <= p[1] <= H + 50:
            cv2.circle(img, p, thickness + 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, p, thickness + 2, (15, 15, 15), 1, lineType=cv2.LINE_AA)
    return img



def read_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def smooth_crop_windows(boxes_xyxy: np.ndarray, margin: float = 1.38) -> np.ndarray:
    """Compute temporally smoothed square crop windows (T, 4) [x0, y0, x1, y1] following skater."""
    cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
    bw = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    bh = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    side = np.maximum(bw, bh) * margin

    def _smooth(arr: np.ndarray, radius: int = 3) -> np.ndarray:
        out = np.empty_like(arr)
        for i in range(len(arr)):
            lo = max(0, i - radius)
            hi = min(len(arr), i + radius + 1)
            out[i] = arr[lo:hi].mean()
        return out

    scx = _smooth(cx, 3)
    scy = _smooth(cy, 3)
    sside = np.maximum(_smooth(side, 5), 64.0)
    half = 0.5 * sside
    return np.stack([scx - half, scy - half, scx + half, scy + half], axis=-1).astype(np.float32)


def extract_square_crop(frame_bgr: np.ndarray, win: np.ndarray, out_size: int) -> tuple[np.ndarray, float, float, float]:
    """Crop square window (x0, y0, x1, y1) with dark border padding and resize to out_size x out_size."""
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = [int(round(float(v))) for v in win]
    side = max(x1 - x0, y1 - y0, 16)
    crop = np.full((side, side, 3), (22, 26, 34), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + side), min(H, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = frame_bgr[sy0:sy1, sx0:sx1]
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    scale = float(out_size) / float(side)
    return resized, float(x0), float(y0), scale



def open_ffmpeg_writer(out_path: Path, width: int, height: int, fps: float, threads: int = 1) -> subprocess.Popen:
    cmd = [
        "nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.3f}",
        "-i", "-",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-threads", str(threads),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)




def compute_metrics(meta: dict, model_key: str, npz_path: Path, boxes_data: dict, fps: float) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    j3d = np.asarray(d["joints3d_cam"], dtype=np.float32)
    j2d = np.asarray(d["joints2d"], dtype=np.float32)
    T = j3d.shape[0]

    pad112 = np.asarray(boxes_data["padded_112"], dtype=np.float32)[:T]
    tight = (boxes_data["tight_clean"] if "tight_clean" in boxes_data else boxes_data["tight"])[:T]

    valid_j = np.all(np.isfinite(j2d), axis=-1)
    in_x = (j2d[:, :, 0] >= pad112[:, 0:1]) & (j2d[:, :, 0] <= pad112[:, 2:3])
    in_y = (j2d[:, :, 1] >= pad112[:, 1:2]) & (j2d[:, :, 1] <= pad112[:, 3:4])
    in_box_pct = float((in_x & in_y & valid_j).mean() * 100.0)

    j_xmin, j_ymin = j2d[:, :, 0].min(axis=1), j2d[:, :, 1].min(axis=1)
    j_xmax, j_ymax = j2d[:, :, 0].max(axis=1), j2d[:, :, 1].max(axis=1)
    j_cx, j_cy = 0.5 * (j_xmin + j_xmax), 0.5 * (j_ymin + j_ymax)
    j_h = np.maximum(j_ymax - j_ymin, 1.0)

    tight_arr = np.asarray([b if b is not None else pad112[i] for i, b in enumerate(tight)], dtype=np.float32)
    t_cx = 0.5 * (tight_arr[:, 0] + tight_arr[:, 2])
    t_cy = 0.5 * (tight_arr[:, 1] + tight_arr[:, 3])
    t_h = np.maximum(tight_arr[:, 3] - tight_arr[:, 1], 1.0)

    center_err_pct = float((np.hypot(j_cx - t_cx, j_cy - t_cy) / t_h).mean() * 100.0)
    h_ratio = float(np.median(j_h / t_h))

    j3d_rel = j3d - j3d[:, 0:1, :]
    if T >= 3:
        accel = j3d_rel[2:] - 2.0 * j3d_rel[1:-1] + j3d_rel[:-2]
        accel_mps2 = float(np.linalg.norm(accel, axis=-1).mean() * (fps ** 2))
    else:
        accel_mps2 = 0.0

    cvs = []
    for a, b in LIMB_EDGES:
        bl = np.linalg.norm(j3d[:, b] - j3d[:, a], axis=-1)
        m = float(bl.mean())
        if m > 1e-4:
            cvs.append(float(bl.std() / m * 100.0))
    bone_cv_pct = float(np.mean(cvs)) if cvs else 0.0

    hip_vec = j3d[:, 4, :] - j3d[:, 1, :]
    yaw = np.arctan2(hip_vec[:, 2], hip_vec[:, 0])
    if T >= 2:
        dyaw = np.rad2deg((np.diff(yaw) + np.pi) % (2.0 * np.pi) - np.pi)
        pelvis_yaw_p95_deg = float(np.percentile(np.abs(dyaw), 95))
        pelvis_abs_turn_deg = float(np.sum(np.abs(dyaw)))
        depth_z_p95_m = float(np.percentile(np.abs(np.diff(j3d[:, 0, 2])), 95))
    else:
        pelvis_yaw_p95_deg = pelvis_abs_turn_deg = depth_z_p95_m = 0.0

    return {
        "sport": meta.get("sport", ""),
        "clip": meta.get("name", ""),
        "width": meta.get("width", 0),
        "height": meta.get("height", 0),
        "frames": T,
        "model": model_key,
        "mesh_type": str(d["mesh_type"]),
        "reproj_in_box_pct": round(in_box_pct, 2),
        "center_err_pct_h": round(center_err_pct, 2),
        "proj_h_ratio": round(h_ratio, 3),
        "accel_jitter_mps2": round(accel_mps2, 2),
        "bone_len_cv_pct": round(bone_cv_pct, 2),
        "pelvis_yaw_p95_deg": round(pelvis_yaw_p95_deg, 2),
        "pelvis_abs_turn_deg": round(pelvis_abs_turn_deg, 1),
        "depth_z_jump_p95_m": round(depth_z_p95_m, 3),
        "fps_infer": round(float(d["fps_infer"]), 2),
        "peak_vram_gb": round(float(d["peak_vram_gb"]), 2),
    }


def render_2d_frame(t, T, meta, frame_bgr, pad112, tight_box, win, preds, labels, cols=4):
    """Render the (cells/cols) x cols 2D reprojection grid for one frame (cell = 360 px).

    Cell 0: full frame with shared boxes; then one cell per model (zoomed follow crop, mesh at alpha 0.40 + H36M-17
    skeleton); last cell: legend. Same look as finefs Stage 2 compare_2d.mp4.
    """
    CELL = 360
    HDR = 26
    IMG_H = CELL - HDR
    cells = []

    c0 = np.full((CELL, CELL, 3), (18, 22, 28), dtype=np.uint8)
    full_draw = frame_bgr.copy()
    H, W = full_draw.shape[:2]
    b112 = pad112[t]
    cv2.rectangle(full_draw, (int(round(b112[0])), int(round(b112[1]))), (int(round(b112[2])), int(round(b112[3]))),
                  (60, 245, 70), max(2, H // 240))
    if tight_box is not None:
        cv2.rectangle(full_draw, (int(round(tight_box[0])), int(round(tight_box[1]))),
                      (int(round(tight_box[2])), int(round(tight_box[3]))), (245, 245, 245), max(1, H // 360))
    scale0 = min(CELL / W, IMG_H / H)
    nw, nh = int(round(W * scale0)), int(round(H * scale0))
    res0 = cv2.resize(full_draw, (nw, nh))
    c0[HDR + (IMG_H - nh) // 2: HDR + (IMG_H - nh) // 2 + nh, (CELL - nw) // 2: (CELL - nw) // 2 + nw] = res0
    cv2.putText(c0, f"Input+Box (f{t}/{T-1})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 240, 248), 1, cv2.LINE_AA)
    cells.append(c0)

    crop_base, wx0, wy0, sc = extract_square_crop(frame_bgr, win, IMG_H)
    for m_key, m_label in labels:
        cell = np.full((CELL, CELL, 3), (18, 22, 28), dtype=np.uint8)
        pdata = preds[m_key]
        K_orig = pdata["K_fullimg"][t]
        K_crop = K_orig.copy()
        K_crop[0, 0] *= sc
        K_crop[1, 1] *= sc
        K_crop[0, 2] = (K_orig[0, 2] - wx0) * sc
        K_crop[1, 2] = (K_orig[1, 2] - wy0) * sc
        patch = crop_base
        if pdata["mesh_type"] != "skeleton":
            patch = render_mesh_painter(crop_base, pdata["verts_cam"][t], pdata["faces"], K_crop, alpha=0.40,
                                        base_bgr=(220, 195, 165))
        else:
            patch = crop_base.copy()
        j2d_crop = (pdata["joints2d"][t] - np.array([wx0, wy0], dtype=np.float32)) * sc
        patch = draw_h36m17_skeleton(patch, j2d_crop, thickness=2)
        px_off = (CELL - IMG_H) // 2
        cell[HDR: HDR + IMG_H, px_off: px_off + IMG_H] = patch
        hdr_col = (155, 225, 255) if pdata["mesh_type"] == "smpl" else ((255, 210, 155) if pdata["mesh_type"] == "mhr" else (170, 255, 170))
        cv2.putText(cell, f"{m_label} {pdata['fps_infer']:.1f}fps", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, hdr_col, 1, cv2.LINE_AA)
        cells.append(cell)

    c7 = np.full((CELL, CELL, 3), (20, 24, 32), dtype=np.uint8)
    box_h = int(round(tight_box[3] - tight_box[1])) if tight_box is not None else int(round(b112[3] - b112[1]))
    info_lines = [
        (f"{meta.get('sport', '')} / {meta.get('name', '')}", (255, 220, 130), 0.50),
        (f"Frame {t} / {T - 1}  ({W}x{H})", (230, 235, 242), 0.48),
        (f"Tight Box H: ~{box_h} px", (200, 225, 245), 0.48),
        ("Input: Green=1.12x  White=Tight", (120, 245, 120), 0.44),
        ("Overlay: Mesh(0.40) + H36M-17", (220, 220, 220), 0.45),
        ("Blue = Left Arm / Left Leg", (255, 155, 75), 0.46),
        ("Red  = Right Arm / Right Leg", (85, 105, 255), 0.46),
        ("Green = Spine & Head", (85, 235, 110), 0.46),
    ]
    for i, (txt, col, fs) in enumerate(info_lines):
        cv2.putText(c7, txt, (14, 48 + i * 36), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    cells.append(c7)

    while len(cells) % cols:
        cells.append(np.full((CELL, CELL, 3), (18, 22, 28), dtype=np.uint8))
    rows = [np.hstack(cells[i: i + cols]) for i in range(0, len(cells), cols)]
    grid = np.vstack(rows)
    for x in range(1, cols):
        cv2.line(grid, (x * CELL, 0), (x * CELL, CELL * len(rows)), (48, 54, 66), 1)
    for r in range(1, len(rows)):
        cv2.line(grid, (0, r * CELL), (CELL * cols, r * CELL), (48, 54, 66), 1)
    return grid


def render_clip(clip_dir: Path, models: list[str] | None = None, workers_ffmpeg_threads: int = 4) -> list[dict]:
    clip = C.Clip(clip_dir)
    boxes_data = json.loads(clip.boxes_json.read_text())
    frames, fps = read_video_frames(clip.mp4)
    T = len(frames)
    pad112 = np.asarray(boxes_data["padded_112"], dtype=np.float32)[:T]
    tight = (boxes_data["tight_clean"] if "tight_clean" in boxes_data else boxes_data["tight"])[:T]
    wins = smooth_crop_windows(pad112, margin=1.40)

    keys = models or [k for k in M.DEFAULT_MODELS + ["rtmpose3d"] if clip.npz(k).exists()]
    keys = [k for k in keys if clip.npz(k).exists()]
    labels = [(k, M.SPECS[k].label) for k in keys]
    preds, rows = {}, []
    for k in keys:
        d = np.load(clip.npz(k), allow_pickle=False)
        verts = np.asarray(d["verts_cam"], dtype=np.float32)[:T]
        faces = np.asarray(d["faces"], dtype=np.int32)
        if str(d["mesh_type"]) != "skeleton":
            faces = ensure_outward_faces(verts, faces)
        preds[k] = {"mesh_type": str(d["mesh_type"]), "verts_cam": verts, "faces": faces,
                    "joints2d": np.asarray(d["joints2d"], dtype=np.float32)[:T],
                    "K_fullimg": np.asarray(d["K_fullimg"], dtype=np.float32)[:T], "fps_infer": float(d["fps_infer"])}
        rows.append(compute_metrics(clip.meta, k, clip.npz(k), boxes_data, fps))

    n_cells = 1 + len(keys) + 1
    cols = 4
    n_rows = (n_cells + cols - 1) // cols
    W_out, H_out = 360 * cols, 360 * n_rows
    mp4 = clip.dir / "compare_2d.mp4"
    proc = open_ffmpeg_writer(mp4, W_out, H_out, fps, threads=workers_ffmpeg_threads)
    for t in range(T):
        grid = render_2d_frame(t, T, clip.meta, frames[t], pad112, tight[t], wins[t], preds, labels, cols)
        proc.stdin.write(grid.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {mp4}")
    with (clip.dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[render] {clip.key}: {len(keys)} models, {T}f -> {mp4}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="hmr_zoo metrics + compare_2d.mp4 for one clip dir")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--models", default="", help="comma-separated (default: every <model>.npz present)")
    a = ap.parse_args()
    render_clip(Path(a.clip), [m for m in a.models.split(",") if m] or None)


if __name__ == "__main__":
    main()
