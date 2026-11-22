"""
gvhmr_fast: GVHMR made fast for long (whole-match) videos, outputting GLOBAL SMPL parameters per video.

Differences from detectors/gvhmr_detector.py (which is untouched; this is a subclass that re-uses its model setup):
  1. ViTPose-H and the HMR2 ViT-H feature extractor run in fp16 (the small GVHMR denoiser stays fp32);
     ViTPose heatmaps are cast back to float32 before the numpy post-processing.
  2. ViTPose flip test is off (flip_test=False).
  3. Person boxes come from preprocessors/track_person.py (--target lock, boxes.json), GVHMR's own YOLO is never run.
  4. The original video file is decoded once through an ffmpeg pipe (same 0.5x scale filter GVHMR uses); nothing is
     re-encoded and no temporary video is written. Crops for ViTPose/HMR2 are cut from those frames in memory.
  5. Frames where the tracker says "no person" (boxes.json "present": false), and frames in segments shorter than
     MIN_SEG, are not run through the 3D model; they are marked absent in the output.
The video is cut into segments (maximal runs of present frames that do not cross a camera shot cut); every segment is
an independent GVHMR sequence with its own SimpleVO camera rotation and its own world frame (see "segment_id").

Output .npz per video (arrays are length T = number of frames, NaN / -1 / False where absent):
  present (T,) bool, segment_id (T,) int (-1 absent), segments (S,3) int [start, end_exclusive, vo_failed],
  smpl_world_{global_orient (T,3), body_pose (T,63), betas (T,10), transl (T,3)}   GVHMR smpl_params_global (Y-up world)
  smpl_cam_{global_orient, body_pose, betas, transl}                              GVHMR smpl_params_incam
  cam_T_w2c (T,4,4)    SimpleVO camera pose per frame (rotation is what GVHMR uses; translation is up to scale);
                       identity when vo_failed. cam_vo_failed (T,) bool: SimpleVO failed, static camera was assumed
  joints24_world (T,24,3), joints24_cam (T,24,3)  SMPL 24 joints from the SMPL body (metres, world Y-up / OpenCV camera)
  joints17_cam (T,17,3)  H36M-17 joints (zoo order) in camera space, for comparison with hmr_zoo npz files
  K (3,3) intrinsics GVHMR assumes (estimate_K), fps, width, height, meta_json (timings, flags, box file)
No mesh vertices are stored. Read with hmr_zoo/fast_io.py (load_fast, to_mstcn_features).

CLI (run in the skeleton_env python, from anywhere):
  python detectors/gvhmr_fast_detector.py --input video.mp4 --boxes boxes.json --out out.npz [--precise]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path = [p for p in sys.path if ".local" not in p] + [p for p in sys.path if ".local" in p]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch

from detectors.gvhmr_detector import GVHMRDetector
from hmr4d.network.hmr2.utils.preproc import IMAGE_MEAN, IMAGE_STD, crop_and_resize
from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
from hmr4d.utils.preproc import Extractor, VitPoseExtractor

MIN_SEG = 10          # frames; shorter segments are not run
VO_STEP = 8           # SimpleVO step, same as GVHMR demo
IMG_DS = 0.5          # GVHMR's own preprocessing scale for ViTPose / HMR2 / SimpleVO
J17_PERM = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # SPIN regressor -> standard H36M order
SKH_ROOT = Path(project_root)


def _skeletonhub_main() -> Path:
    return Path(os.environ.get("SKELETONHUB_ROOT", "/home/allen/SkeletonHub"))


class GVHMRFastDetector(GVHMRDetector):
    def __init__(self, device="cuda:0", fp16=True, flip_test=False, vo_workers=4):
        super().__init__(device=device)
        self.vp = VitPoseExtractor()
        self.hmr2 = Extractor()
        self.vo_workers = vo_workers
        self.set_mode(fp16, flip_test)
        j17 = np.load(_skeletonhub_main() / "common_models/body_models/smpl/J_regressor_h36m.npy").astype(np.float32)[J17_PERM]
        self.J17 = torch.from_numpy(j17).to(device)
        self.mean = IMAGE_MEAN.clone().float().to(device).view(1, 3, 1, 1)
        self.std = IMAGE_STD.clone().float().to(device).view(1, 3, 1, 1)

    def set_mode(self, fp16: bool, flip_test: bool):
        self.fp16, self.flip_test = fp16, flip_test
        self.vp.flip_test = flip_test
        dt = torch.float16 if fp16 else torch.float32
        self.dtype = dt
        self.vp.pose.to(dt)
        self.hmr2.extractor.backbone.to(dt)          # ViT-H backbone in fp16; the small SMPL token head stays fp32
        self.hmr2.extractor.smpl_head.to(torch.float32)

    # ------------------------------------------------------------------------------------------ decoding
    @staticmethod
    def _open_ffmpeg(video, W, H):
        w2, h2 = int(W * IMG_DS), int(H * IMG_DS)
        cmd = ["ffmpeg", "-loglevel", "error", "-i", str(video), "-an", "-vsync", "0", "-vf",
               f"scale=iw*{IMG_DS}:ih*{IMG_DS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w2 * h2 * 3 * 4), w2, h2

    # ------------------------------------------------------------------------------------------ network stages
    def _norm(self, crops_u8):
        x = torch.from_numpy(crops_u8).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return ((x - self.mean) / self.std).to(self.dtype)

    @torch.no_grad()
    def _vitpose(self, crops, bbx_xys, bs=32):
        from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17

        out = []
        for j in range(0, len(crops), bs):
            x = self._norm(crops[j:j + bs])[:, :, :, 32:224]
            if self.flip_test:
                hm, hmf = self.vp.pose(torch.cat([x, x.flip(3)], dim=0)).chunk(2)
                hm = (hm.float() + flip_heatmap_coco17(hmf.float())) * 0.5
            else:
                hm = self.vp.pose(x).float()                       # fp16 -> float32 before numpy post-processing
            b = bbx_xys[j:j + bs]
            center = b[:, :2].numpy()
            scale = (torch.cat((b[:, [2]] * 24 / 32, b[:, [2]]), dim=1) / 200).numpy()
            preds, maxvals = keypoints_from_heatmaps(heatmaps=hm.cpu().numpy(), center=center, scale=scale, use_udp=True)
            out.append(torch.from_numpy(np.concatenate((preds, maxvals), axis=-1)))
        return torch.cat(out, 0)

    @torch.no_grad()
    def _hmr2(self, crops, bs=32):
        out = []
        for j in range(0, len(crops), bs):
            ex = self.hmr2.extractor
            x = self._norm(crops[j:j + bs])[:, :, :, 32:-32]     # same as HMR2.forward(feat_mode=True)
            f = ex.smpl_head(ex.backbone(x).float(), only_return_token_out=True)
            out.append(f.float().cpu())
        return torch.cat(out, 0)

    # ------------------------------------------------------------------------------------------ camera
    def _vo(self, frames, sample_idxs):
        """SimpleVO (SIFT matches + pycolmap two-view pose between every 8th frame), made failure-tolerant: GVHMR's own
        SimpleVO aborts the whole sequence when a single frame pair cannot be solved (pycolmap returns None) and the demo then
        falls back to a static camera for the entire video. Here a failed pair counts as 'no camera motion' for that pair
        and the sequence only falls back to a static camera when more than half of the pairs fail.
        Returns (T_w2c (L,4,4), failed_pair_fraction)."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from hmr4d.utils.preproc.relpose.matcher_wrapper import Matcher
        from hmr4d.utils.preproc.relpose.solver_two_view import CameraParams, TwoPairSolver, interpolate_missing_frames
        from hmr4d.utils.preproc.relpose.utils import focal_length_from_mm

        F, H, W, _ = frames.shape
        matcher = Matcher("sift")
        solver = TwoPairSolver(CameraParams(W, H, focal_length=focal_length_from_mm(W, H, 24)), solver="pycolmap")
        tls = threading.local()

        def pair(i):
            if not hasattr(tls, "m"):
                tls.m, tls.s = matcher.clone(), solver.clone()
            try:
                p0, p1 = tls.m.match_np(frames[i], frames[i + 1])
                T = tls.s.solve(p0, p1)
                if T is None or not np.all(np.isfinite(T)):
                    raise ValueError("no pose")
                return np.asarray(T, dtype=np.float64), False
            except Exception:
                return np.eye(4), True

        with ThreadPoolExecutor(max_workers=max(1, self.vo_workers)) as ex:
            res = list(ex.map(pair, range(F - 1)))
        T_list = [np.eye(4)]
        for T, _ in res:
            T_list.append(T @ T_list[-1])
        fail = float(np.mean([f for _, f in res])) if res else 0.0
        return np.stack(interpolate_missing_frames(T_list, sample_idxs)).astype(np.float32), fail

    # ------------------------------------------------------------------------------------------ joints from SMPL
    @torch.no_grad()
    def _joints(self, params, want17=False, chunk=128):
        j24, j17 = [], []
        for i in range(0, params["body_pose"].shape[0], chunk):
            p = {k: v[i:i + chunk].to(self.device) for k, v in params.items()}
            o = self.smplx(**p)
            verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in o.vertices])
            j24.append(torch.matmul(self.J_regressor_24, verts).cpu())
            if want17:
                j17.append(torch.matmul(self.J17, verts).cpu())
        return torch.cat(j24).numpy(), (torch.cat(j17).numpy() if want17 else None)

    # ------------------------------------------------------------------------------------------ main entry
    def detect_video_fast(self, video_path, boxes_json, key_box="padded_112", static_cam=False, verbose=True):
        t_all = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        bj = json.loads(Path(boxes_json).read_text())
        W, H = int(bj["width"]), int(bj["height"])
        fps = bj.get("fps")
        if not fps:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
        fps = float(fps)
        n = int(bj["n_frames"])
        present = np.asarray(bj.get("present", [True] * n), dtype=bool)[:n]
        n = len(present)
        boxes = np.asarray(bj[key_box], dtype=np.float32) if key_box in bj else np.zeros((n, 4), np.float32)
        cuts = {int(c[0]) if isinstance(c, (list, tuple)) else int(c) for c in bj.get("shot_cuts", [])}

        # segments: maximal present runs not crossing a shot cut, at least MIN_SEG long
        seg_of = -np.ones(n, dtype=np.int32)
        segs = []
        t = 0
        while t < n:
            if not present[t]:
                t += 1
                continue
            s = t
            t += 1
            while t < n and present[t] and t not in cuts:
                t += 1
            if t - s >= MIN_SEG:
                seg_of[s:t] = len(segs)
                segs.append([s, t])
        use = seg_of >= 0
        tim = {}

        # --- decode once, crop in memory
        t0 = time.perf_counter()
        bxy = torch.zeros(n, 3)
        if use.any():
            bxy[torch.from_numpy(use)] = get_bbx_xys_from_xyxy(torch.from_numpy(boxes[use]), base_enlarge=1.2).float()
        crops = np.zeros((int(use.sum()), 256, 256, 3), np.uint8)
        crop_row = -np.ones(n, dtype=np.int64)
        crop_row[use] = np.arange(int(use.sum()))
        vo_need = {}
        for si, (a, b) in enumerate(segs):
            for tt in sorted(set(range(a, b, VO_STEP)) | {b - 1}):
                vo_need[tt] = si
        vo_frames = {}
        proc, w2, h2 = self._open_ffmpeg(video_path, W, H)
        fb = w2 * h2 * 3
        n_dec = 0
        while n_dec < n:
            buf = proc.stdout.read(fb)
            if len(buf) < fb:
                break
            if use[n_dec] or n_dec in vo_need:
                fr = np.frombuffer(buf, np.uint8).reshape(h2, w2, 3)
                if use[n_dec]:
                    size_ds = float(bxy[n_dec, 2]) * IMG_DS
                    d = size_ds / 256 / 2.0
                    src = cv2.GaussianBlur(fr, (5, 5), (d - 1) / 2) if d > 1.1 else fr   # GVHMR anti-aliasing blur
                    c, _ = crop_and_resize(src, bxy[n_dec, :2].numpy() * IMG_DS, size_ds, 256, enlarge_ratio=1.0)
                    crops[crop_row[n_dec]] = c
                if n_dec in vo_need:
                    vo_frames[n_dec] = fr.copy()
            n_dec += 1
        proc.stdout.close()
        proc.terminate()
        proc.wait()
        if n_dec < n:
            print(f"[gvhmr_fast] WARNING: video decoded {n_dec} frames, boxes cover {n}", flush=True)
        tim["decode_crop"] = time.perf_counter() - t0

        K = estimate_K(W, H).float()
        S = len(segs)
        out = {}
        flds = ("global_orient", "body_pose", "betas", "transl")
        shp = {"global_orient": 3, "body_pose": 63, "betas": 10, "transl": 3}
        for pre in ("smpl_world_", "smpl_cam_"):
            for k in flds:
                out[pre + k] = np.full((n, shp[k]), np.nan, np.float32)
        out["joints24_world"] = np.full((n, 24, 3), np.nan, np.float32)
        out["joints24_cam"] = np.full((n, 24, 3), np.nan, np.float32)
        out["joints17_cam"] = np.full((n, 17, 3), np.nan, np.float32)
        T_w2c = np.full((n, 4, 4), np.nan, np.float32)
        vo_failed_seg = np.zeros(S, bool)
        vo_errors = []
        vo_fail_frac = []
        tim.update(vitpose=0.0, hmr2=0.0, vo=0.0, gvhmr=0.0, joints=0.0)

        for si, (a, b) in enumerate(segs):
            rows = crop_row[a:b]
            seg_crops = crops[rows]
            seg_bxy = bxy[a:b]
            L = b - a
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            kp2d = self._vitpose(seg_crops, seg_bxy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            feats = self._hmr2(seg_crops)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            # camera
            use_static = static_cam
            if not static_cam:
                try:
                    idxs = np.array(sorted(set(range(0, L, VO_STEP)) | {L - 1}))
                    fr = np.stack([vo_frames[a + int(i)] for i in idxs])
                    T, fail = self._vo(fr, idxs)
                    vo_fail_frac.append(round(fail, 3))
                    if fail > 0.5:
                        raise RuntimeError(f"{fail:.0%} of frame pairs failed")
                except Exception as exc:  # SimpleVO can fail on tiny motion; GVHMR falls back to a static camera
                    vo_errors.append(f"seg{si}: {type(exc).__name__}: {exc}"[:300])
                    if verbose:
                        print(f"[gvhmr_fast] SimpleVO failed on segment {si} ({exc}); static camera assumed", flush=True)
                    use_static = True
            if use_static:
                T = np.tile(np.eye(4, dtype=np.float32), (L, 1, 1))
                vo_failed_seg[si] = not static_cam
            R_w2c = torch.from_numpy(T[:, :3, :3]).float()
            t3 = time.perf_counter()
            data = {"length": torch.tensor(L), "bbx_xys": seg_bxy, "kp2d": kp2d, "K_fullimg": K.repeat(L, 1, 1),
                    "cam_angvel": compute_cam_angvel(R_w2c), "f_imgseq": feats}
            pred = self.model.predict(data, static_cam=use_static)
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            pg = {k: v.detach() for k, v in pred["smpl_params_global"].items()}
            pc = {k: v.detach() for k, v in pred["smpl_params_incam"].items()}
            jw, _ = self._joints(pg)
            jc, j17 = self._joints(pc, want17=True)
            T_w2c[a:b] = T
            out["joints24_world"][a:b], out["joints24_cam"][a:b], out["joints17_cam"][a:b] = jw, jc, j17
            for k in flds:
                out["smpl_world_" + k][a:b] = pg[k].cpu().numpy()
                out["smpl_cam_" + k][a:b] = pc[k].cpu().numpy()
            t5 = time.perf_counter()
            for k, v in (("vitpose", t1 - t0), ("hmr2", t2 - t1), ("vo", t3 - t2), ("gvhmr", t4 - t3), ("joints", t5 - t4)):
                tim[k] += v
            if verbose:
                print(f"[gvhmr_fast] segment {si + 1}/{S} frames {a}-{b} ({L}) vo_failed={vo_failed_seg[si]}", flush=True)
            del pred, pg, pc

        total = time.perf_counter() - t_all
        torch.cuda.synchronize()
        meta = {"model": "gvhmr_fast", "fp16": self.fp16, "flip_test": self.flip_test, "box_key": key_box,
                "static_cam": static_cam, "timing_sec": {k: round(v, 2) for k, v in tim.items()},
                "total_sec": round(total, 2), "n_frames": int(n), "n_present_input": int(np.sum(bj.get("present", [True] * n))),
                "n_run": int(use.sum()), "n_segments": S, "n_vo_failed_segments": int(vo_failed_seg.sum()), "vo_errors": vo_errors[:10], "vo_pair_fail_frac": vo_fail_frac,
                "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                "min_seg": MIN_SEG, "video": str(video_path), "boxes": str(boxes_json)}
        return dict(
            present=use, segment_id=seg_of.astype(np.int16),
            segments=np.array([[a, b, int(vo_failed_seg[i])] for i, (a, b) in enumerate(segs)], dtype=np.int32).reshape(-1, 3),
            cam_T_w2c=T_w2c, cam_vo_failed=np.array([bool(vo_failed_seg[s]) if s >= 0 else False for s in seg_of]),
            K=K.numpy(), fps=np.float32(fps), width=np.int32(W), height=np.int32(H),
            meta_json=np.asarray(json.dumps(meta)), **out)


def save_fast_npz(path, res):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **res)


def main():
    ap = argparse.ArgumentParser(description="gvhmr_fast: fp16 GVHMR on a locked-person box track -> global SMPL npz")
    ap.add_argument("--input", required=True)
    ap.add_argument("--boxes", required=True, help="boxes.json from preprocessors/track_person.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precise", action="store_true", help="fp32 + ViTPose flip test (the original GVHMR numerics)")
    ap.add_argument("--static-cam", action="store_true")
    a = ap.parse_args()
    det = GVHMRFastDetector(a.device, fp16=not a.precise, flip_test=a.precise)
    res = det.detect_video_fast(a.input, a.boxes, static_cam=a.static_cam)
    save_fast_npz(a.out, res)
    print(json.loads(str(res["meta_json"])))


if __name__ == "__main__":
    main()
