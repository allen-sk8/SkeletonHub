"""
Optimized HybrIK Prediction Service.
Adapted from original CoachMe Service for local execution in SkeletonHub.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
from easydict import EasyDict as edict
from torchvision import transforms as T
from collections import defaultdict
import pickle
import subprocess
import json
import time

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_max_iou_box, get_one_box


class HybrikFastRequest(BaseModel):
    video_path: str
    output_dir: Optional[str] = "runs/hybrik"
    save_name: str = "skeleton"
    gpu: int = 0
    flip_test: bool = True
    skip_detection: bool = True
    batch_size: int = 1
    save_vertices: bool = False  # Save SMPL vertices (6890x3) for mesh rendering
    bbox_file: Optional[str] = None


class HybrikFastResponse(BaseModel):
    input_video_path: str
    output_pkl_path: str
    message: str
    total_frames: int
    elapsed_seconds: float
    timing_breakdown: Optional[Dict[str, float]] = None


# ────────────────────────────────────────────────────────────────────────────────
# Global Translation Utils
# ────────────────────────────────────────────────────────────────────────────────

def read_bbox_centers(txt_path: Path) -> List[Tuple[float, float, float]]:
    """Parse YOLO bbox txt, return per-frame (cx, cy, h). Expects x1 y1 x2 y2 format."""
    raw_boxes = []
    with open(txt_path) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 4:
                raw_boxes.append([float(p) for p in parts[:4]])
            else:
                raw_boxes.append([-1.0, -1.0, -1.0, -1.0])
                
    # Forward fill
    last_valid = None
    for idx in range(len(raw_boxes)):
        box = raw_boxes[idx]
        if box[0] == -1.0 or box[2] - box[0] <= 0:
            if last_valid is not None:
                raw_boxes[idx] = last_valid.copy()
        else:
            last_valid = box.copy()
            
    # Backward fill
    if last_valid is not None:
        for idx in range(len(raw_boxes)):
            box = raw_boxes[idx]
            if box[0] == -1.0 or box[2] - box[0] <= 0:
                for k in range(idx + 1, len(raw_boxes)):
                    if raw_boxes[k][0] != -1.0 and raw_boxes[k][2] - raw_boxes[k][0] > 0:
                        raw_boxes[idx] = raw_boxes[k].copy()
                        break

    centers = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box
        if x1 == -1.0:
            cx, cy, h = 0.0, 0.0, 1.0
        else:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            h = abs(y2 - y1)
        centers.append((cx, cy, h))
    return centers


def apply_translation(skel: np.ndarray, centers: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Apply vertical global translation from YOLO bbox to skeleton.
    Uses bbox cy (vertical center) to estimate global vertical displacement,
    compensating for the pelvis-relative coordinate system of HybrIK.
    """
    if not centers:
        return skel

    # Compute per-frame bbox height
    heights = [h for _, _, h in centers]
    avg_height = np.mean(heights)

    # Scale factor based on assumed reference height (adjustable per dataset)
    base_height = 300.0
    scale = np.clip(base_height / avg_height, 0.6, 1.4)

    skel = skel.copy()
    T = skel.shape[0]
    for t in range(min(T, len(centers))):
        _, cy, _ = centers[t]
        skel[t, :, 1] += cy / 300 * scale

    # Zero out first-frame pelvis to keep only relative motion
    skel -= skel[0, 0, :]
    return skel


# ────────────────────────────────────────────────────────────────────────────────
# Video FPS
# ────────────────────────────────────────────────────────────────────────────────

def get_video_fps_ffprobe(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-print_format", "json",
        "-show_entries", "stream=r_frame_rate",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return 30.0
        info = json.loads(result.stdout)
        if not info or 'streams' not in info or not info['streams']:
            return 30.0
        r_frame_rate = info['streams'][0]['r_frame_rate']
        num, denom = map(int, r_frame_rate.split('/'))
        return num / denom if denom != 0 else 30.0
    except Exception:
        return 30.0


# ────────────────────────────────────────────────────────────────────────────────
# Service
# ────────────────────────────────────────────────────────────────────────────────

class HybrikServiceFast:
    """Optimized HybrIK inference service with batch support."""

    def __init__(self, gpu=0, skip_detection=True):
        self.gpu = gpu
        self.skip_detection = skip_detection
        
        # Paths configured to use local workspace assets
        project_root = Path(__file__).resolve().parent.parent.parent
        self.cfg_file = str(project_root / 'external/Hybrik/configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml')
        self.ckpt_path = str(project_root / 'common_models/checkpoints/hybrik/hybrik_hrnet.pth')
        
        self._initialize_models()
        print(f'[ServiceFast] Initialized. skip_detection={skip_detection}, ckpt={self.ckpt_path}')

    def _initialize_models(self):
        self.det_transform = T.Compose([T.ToTensor()])

        if not self.skip_detection:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.det_model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.det_model.cuda(self.gpu)
            self.det_model.eval()
            print('[ServiceFast] Faster R-CNN loaded.')
        else:
            self.det_model = None
            print('[ServiceFast] Skipping Faster R-CNN (using full-frame bbox).')

        cfg = update_config(self.cfg_file)
        self.cfg = cfg

        bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
        dummy_set = edict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
        })

        self.transformation = SimpleTransform3DSMPLCam(
            dummy_set,
            scale_factor=cfg.DATASET.SCALE_FACTOR,
            color_factor=cfg.DATASET.COLOR_FACTOR,
            occlusion=cfg.DATASET.OCCLUSION,
            input_size=cfg.MODEL.IMAGE_SIZE,
            output_size=cfg.MODEL.HEATMAP_SIZE,
            depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=bbox_3d_shape,
            rot=cfg.DATASET.ROT_FACTOR,
            sigma=cfg.MODEL.EXTRA.SIGMA,
            train=False,
            add_dpg=False,
            loss_type=cfg.LOSS['TYPE']
        )

        self.hybrik_model = builder.build_sppe(cfg.MODEL)
        print(f'[ServiceFast] Loading HybrIK from {self.ckpt_path}...)')
        save_dict = torch.load(self.ckpt_path, map_location='cpu')
        if type(save_dict) == dict:
            model_dict = save_dict['model']
            self.hybrik_model.load_state_dict(model_dict)
        else:
            self.hybrik_model.load_state_dict(save_dict)

        self.hybrik_model.cuda(self.gpu)
        self.hybrik_model.eval()

    def _detect_person(self, input_image, prev_box):
        """Run Faster R-CNN detection on a single frame."""
        det_input = self.det_transform(input_image).to(self.gpu)
        det_output = self.det_model([det_input])[0]
        if prev_box is None:
            tight_bbox = get_one_box(det_output)
        else:
            tight_bbox = get_max_iou_box(det_output, prev_box)
        return tight_bbox

    def _extract_batch_results(self, pose_output, batch_size, res_db, heights, widths, bboxes_np, save_vertices=False):
        """Extract results from a batched model output into res_db lists."""
        for j in range(batch_size):
            res_db['pred_xyz_24_struct'].append(
                pose_output.pred_xyz_jts_24_struct[j].reshape(24, 3).cpu().data.numpy())
            res_db['pred_xyz_17'].append(
                pose_output.pred_xyz_jts_17[j].reshape(17, 3).cpu().data.numpy())
            res_db['pred_xyz_29'].append(
                pose_output.pred_xyz_jts_29[j].reshape(29, 3).cpu().data.numpy())
            res_db['pred_uvd'].append(
                pose_output.pred_uvd_jts[j].reshape(29, 3).cpu().data.numpy())
            res_db['pred_scores'].append(
                pose_output.maxvals[j, :29].reshape(29).cpu().data.numpy())
            res_db['pred_betas'].append(
                pose_output.pred_shape[j].cpu().data.numpy())
            res_db['pred_thetas'].append(
                pose_output.pred_theta_mats[j].cpu().data.numpy())
            res_db['pred_phi'].append(
                pose_output.pred_phi[j].cpu().data.numpy())
            res_db['pred_camera'].append(
                pose_output.pred_camera[j].cpu().data.numpy())
            res_db['pred_cam_root'].append(
                pose_output.cam_root[j].cpu().numpy())
            res_db['transl'].append(
                pose_output.transl[j].cpu().data.numpy())
            if save_vertices:
                res_db['pred_vertices'].append(
                    pose_output.pred_vertices[j].cpu().data.numpy())
            res_db['bbox'].append(bboxes_np[j])
            res_db['height'].append(heights[j])
            res_db['width'].append(widths[j])

    def process_video(self, request: HybrikFastRequest) -> HybrikFastResponse:
        """Process video with batching and per-step timing."""
        timings = defaultdict(float)

        video_path = Path(request.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {request.video_path}")

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.save_name}.pk"

        batch_size = request.batch_size

        res_keys = [
            'pred_xyz_24_struct', 'pred_xyz_17', 'pred_xyz_29', 'pred_uvd',
            'pred_scores', 'pred_betas', 'pred_thetas', 'pred_phi',
            'pred_camera', 'pred_cam_root', 'transl', 'bbox', 'height', 'width',
        ]
        if request.save_vertices:
            res_keys.append('pred_vertices')
        res_db = {k: [] for k in res_keys}

        # ── 1. Read all frames ──────────────────────────────────────
        t = time.time()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = get_video_fps_ffprobe(str(video_path))

        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()
        timings['1_video_read'] = time.time() - t

        total_frames = len(raw_frames)
        if total_frames == 0:
            raise RuntimeError("Video has 0 frames")

        # Load precomputed bboxes for cropping if available
        bbox_list_all = []
        bbox_file_to_load = None
        if request.bbox_file:
            bbox_file_to_load = Path(request.bbox_file)
            if not bbox_file_to_load.exists():
                bbox_file_to_load = None
        if bbox_file_to_load is None:
            yolo_dir = video_path.parent / "yolo"
            bbox_txt = yolo_dir / "video.txt"
            if bbox_txt.exists():
                bbox_file_to_load = bbox_txt
            else:
                fallback_1 = yolo_dir / f"{video_path.stem}.txt"
                if fallback_1.exists():
                    bbox_file_to_load = fallback_1
                else:
                    clean_stem = video_path.stem.replace('_output', '').replace('_yolo', '')
                    fallback_2 = yolo_dir / f"{clean_stem}.txt"
                    if fallback_2.exists():
                        bbox_file_to_load = fallback_2
                    else:
                        if yolo_dir.exists():
                            txt_files = list(yolo_dir.glob("*.txt"))
                            if txt_files:
                                bbox_file_to_load = txt_files[0]
                                
        if bbox_file_to_load is not None and bbox_file_to_load.exists():
            print(f"[ServiceFast] 📦 Loading bboxes for cropping from: {bbox_file_to_load}")
            with open(bbox_file_to_load, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        bbox_list_all.append([float(p) for p in parts[:4]])
                    else:
                        bbox_list_all.append([-1.0, -1.0, -1.0, -1.0])
            
            # Fill missing frames (-1) using forward/backward carryover
            last_valid = None
            for idx in range(len(bbox_list_all)):
                box = bbox_list_all[idx]
                if box[0] == -1.0 or box[2] - box[0] <= 0:
                    if last_valid is not None:
                        bbox_list_all[idx] = last_valid.copy()
                else:
                    last_valid = box.copy()
            if last_valid is not None:
                for idx in range(len(bbox_list_all)):
                    box = bbox_list_all[idx]
                    if box[0] == -1.0 or box[2] - box[0] <= 0:
                        for k in range(idx + 1, len(bbox_list_all)):
                            if bbox_list_all[k][0] != -1.0 and bbox_list_all[k][2] - bbox_list_all[k][0] > 0:
                                bbox_list_all[idx] = bbox_list_all[k].copy()
                                break
        else:
            print("[ServiceFast] ⚠️ No precomputed bbox file found for cropping, using full-frame")

        # ── 2. Process in batches (with Adaptive OOM Retry) ──────────
        frame_idx = 0
        current_batch_size = batch_size
        frame_count = 0

        while frame_idx < total_frames:
            bs = min(current_batch_size, total_frames - frame_idx)
            batch_frames = raw_frames[frame_idx : frame_idx + bs]

            # ── 2a. BGR→RGB + bbox ──
            t = time.time()
            rgb_images = []
            heights, widths = [], []
            for frame in batch_frames:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_images.append(img)
                heights.append(img.shape[0])
                widths.append(img.shape[1])
            t_convert = time.time() - t

            # ── 2b. Transform (crop/resize/normalize) ──
            t = time.time()
            pose_inputs, bboxes_list, centers_list = [], [], []
            for i, (img, h, w) in enumerate(zip(rgb_images, heights, widths)):
                global_frame_idx = frame_idx + i
                tight_bbox = None
                if global_frame_idx < len(bbox_list_all):
                    box = bbox_list_all[global_frame_idx]
                    if box[0] != -1.0:
                        tight_bbox = box
                
                if tight_bbox is None:
                    tight_bbox = [0, 0, w, h]

                pose_input, bbox, img_center = self.transformation.test_transform(img, tight_bbox)
                pose_inputs.append(pose_input)
                bboxes_list.append(np.array(bbox))
                centers_list.append(img_center)
            t_transform = time.time() - t

            try:
                # ── 2c. GPU transfer ──
                t = time.time()
                pose_batch = torch.stack(pose_inputs).to(self.gpu)
                bboxes_t = torch.from_numpy(np.array(bboxes_list)).to(self.gpu).float()
                centers_t = torch.from_numpy(np.array(centers_list)).to(self.gpu).float()
                torch.cuda.synchronize()
                t_transfer = time.time() - t

                # ── 2d. Model forward ──
                t = time.time()
                with torch.no_grad():
                    pose_output = self.hybrik_model(
                        pose_batch,
                        flip_test=request.flip_test,
                        bboxes=bboxes_t,
                        img_center=centers_t
                    )
                torch.cuda.synchronize()
                t_forward = time.time() - t

                # ── 2e. Result extraction (GPU→CPU) ──
                t = time.time()
                self._extract_batch_results(
                    pose_output, bs, res_db, heights, widths, bboxes_list,
                    save_vertices=request.save_vertices)
                t_extract = time.time() - t

                # Accumulate times on success
                timings['2_color_convert'] += t_convert
                timings['3_transform'] += t_transform
                timings['4_gpu_transfer'] += t_transfer
                timings['5_model_forward'] += t_forward
                timings['6_result_extract'] += t_extract

                frame_idx += bs
                frame_count += bs

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if current_batch_size <= 1:
                        raise e
                    old_bs = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"\n⚠️  [ServiceFast] CUDA Out of Memory detected at batch_size={old_bs}.")
                    print(f"   📉 Automatically reducing batch_size to {current_batch_size} and retrying frame {frame_idx}...\n")
                    # Clean local references to free GPU memory
                    if 'pose_inputs' in locals(): del pose_inputs
                    if 'bboxes_list' in locals(): del bboxes_list
                    if 'centers_list' in locals(): del centers_list
                    if 'pose_batch' in locals(): del pose_batch
                    if 'bboxes_t' in locals(): del bboxes_t
                    if 'centers_t' in locals(): del centers_t
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # ── 3. Stack results ────────────────────────────────────────
        t = time.time()
        if frame_count > 0:
            for k in res_db:
                res_db[k] = np.stack(res_db[k]) if isinstance(res_db[k][0], np.ndarray) else np.array(res_db[k])

        # ── 4. Compute global translation from YOLO bbox ────────────
        # ── 4. Compute global translation from YOLO bbox ────────────
        bbox_txt = None
        if request.bbox_file:
            bbox_txt = Path(request.bbox_file)
            if not bbox_txt.exists():
                print(f"[ServiceFast] ⚠️ Provided bbox_file does not exist: {request.bbox_file}")
                bbox_txt = None

        if bbox_txt is None:
            yolo_dir = video_path.parent / "yolo"
            bbox_txt = yolo_dir / "video.txt"
            
            # Try fallbacks if video.txt doesn't exist
            if not bbox_txt.exists():
                # Fallback 1: video_stem.txt (e.g. yolo_output.txt)
                fallback_1 = yolo_dir / f"{video_path.stem}.txt"
                if fallback_1.exists():
                    bbox_txt = fallback_1
                else:
                    # Fallback 2: strip _output or similar suffixes
                    clean_stem = video_path.stem.replace('_output', '').replace('_yolo', '')
                    fallback_2 = yolo_dir / f"{clean_stem}.txt"
                    if fallback_2.exists():
                        bbox_txt = fallback_2
                    else:
                        # Fallback 3: First available .txt file in yolo/
                        if yolo_dir.exists():
                            txt_files = list(yolo_dir.glob("*.txt"))
                            if txt_files:
                                bbox_txt = txt_files[0]

        if bbox_txt.exists():
            centers = read_bbox_centers(bbox_txt)
            global_struct = apply_translation(
                res_db['pred_xyz_24_struct'].copy(), centers)
            print(f'[ServiceFast] Applied global translation from {bbox_txt} ({len(centers)} bbox frames)')
        else:
            global_struct = res_db['pred_xyz_24_struct']
            print(f'[ServiceFast] No YOLO bbox txt found at {bbox_txt or (yolo_dir / "video.txt")}, using relative coords as global')

        # ── 5. Write skeleton.pk (backward-compatible) ──────────────
        skeleton_result = {
            "video_name": video_path.stem,
            "features": res_db['pred_xyz_24_struct'],           # pelvis-relative (CoachMe)
            "pred_xyz_24_struct_global": global_struct,          # global translation (ReasonMotion)
            "fps": fps,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(skeleton_result, f)

        # ── 6. Write smpl.pk (full SMPL parameters) ─────────────────
        smpl_path = output_dir / "smpl.pk"
        smpl_keys = [
            'pred_betas', 'pred_thetas', 'pred_phi', 'pred_camera',
            'pred_cam_root', 'transl', 'pred_xyz_17', 'pred_xyz_29',
            'pred_uvd', 'pred_scores', 'bbox',
        ]
        smpl_result = {k: res_db[k] for k in smpl_keys}
        smpl_result['fps'] = fps
        if request.save_vertices and 'pred_vertices' in res_db:
            smpl_result['pred_vertices'] = res_db['pred_vertices']
        with open(smpl_path, 'wb') as f:
            pickle.dump(smpl_result, f)

        timings['7_pickle_write'] = time.time() - t

        # ── Total ──
        timings['total'] = sum(timings.values())

        print(f'[ServiceFast] Done: {frame_count} frames in {timings["total"]:.2f}s '
              f'({frame_count/timings["total"]:.1f} FPS), '
              f'flip={request.flip_test}, bs={batch_size}')
        print(f'[ServiceFast] Output: skeleton.pk={output_path}, smpl.pk={smpl_path}')

        return HybrikFastResponse(
            input_video_path=str(video_path),
            output_pkl_path=str(output_path),
            message=f"Processed {frame_count} frames in {timings['total']:.2f}s",
            total_frames=frame_count,
            elapsed_seconds=round(timings['total'], 3),
            timing_breakdown=dict(timings)
        )
