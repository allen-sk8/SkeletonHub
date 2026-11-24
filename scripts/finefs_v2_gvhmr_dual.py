"""FineFS v2 test, GVHMR stage with two crop resolutions in one pass (run in skeleton_env, one GPU).

For every video: decode the original file once at full resolution and once through GVHMR's 0.5x scale filter (two ffmpeg
pipes read in lock step), then for every segment (same segment rule as detectors/gvhmr_fast_detector.py):
  * SimpleVO camera rotation ONCE, on the 0.5x frames (as gvhmr_fast does), shared by both variants, so the two variants
    differ only in where the person crop comes from;
  * variant "half": crops cut from the 0.5x frames  = exactly gvhmr_fast (fp16, no flip test);
  * variant "full": crops cut from the full-resolution frames (same box, same 256x256 crop, anti-alias blur scaled to the
    source resolution).
  Each variant: ViTPose-H 2D (kept and saved), HMR2 features, GVHMR, SMPL joints.

Writes (out = outputs/finefs_v2_test):
  out/3_gvhmr/<id>/gvhmr_fast.npz            variant half (the current default), same format as gvhmr_fast_detector.py
  out/2_pose2d/<id>/vitpose_h.npz            kp2d (T,17,3) of variant half [x, y, conf] in original-video pixels, NaN absent
  out/_exp3_resolution/<id>/gvhmr_full.npz   variant full, same format; pose2d_full.npz the full-res 2D; stats.json timings
  --variants half: production run, no full-resolution decode; stats.json goes to out/3_gvhmr/<id>/

  python scripts/finefs_v2_gvhmr_dual.py --ids 36,261 --boxes-root outputs/finefs_v2_test/1_tracking --out outputs/finefs_v2_test
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--boxes-root", required=True, help="dir with <id>/boxes.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default="/home/allen/datasets/FineFS/video")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variants", default="half,full", help="'half' alone = production run (no full-resolution decode)")
    ap.add_argument("--fail-dir", default=None, help="record per-video failures here and go on (see finefs_failguard.py)")
    a = ap.parse_args()
    from finefs_failguard import FailGuard
    guard = FailGuard(a.fail_dir, "gvhmr")
    variants = [v for v in ("half", "full") if v in a.variants.split(",")]
    do_full = "full" in variants
    out = Path(a.out).resolve()
    boxes_root = Path(a.boxes_root).resolve()
    os.chdir(str(ROOT / "external" / "gvhmr"))
    import cv2
    import numpy as np
    import torch

    from detectors.gvhmr_fast_detector import IMG_DS, MIN_SEG, VO_STEP, GVHMRFastDetector
    from hmr4d.network.hmr2.utils.preproc import crop_and_resize
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr_zoo.vram import ProcessVramMonitor

    base = ProcessVramMonitor().baseline_mb
    det = GVHMRFastDetector(a.device, fp16=True, flip_test=False)

    def crop_half(fr, c, s):
        size_ds = s * IMG_DS
        d = size_ds / 256 / 2.0
        src = cv2.GaussianBlur(fr, (5, 5), (d - 1) / 2) if d > 1.1 else fr       # identical to gvhmr_fast
        return crop_and_resize(src, c * IMG_DS, size_ds, 256, enlarge_ratio=1.0)[0]

    def crop_full(fr, c, s):
        d = s / 256 / 2.0
        H, W = fr.shape[:2]
        r = int(np.ceil(s / 2 + 8))                                              # blur only the region around the crop
        x0, y0 = max(0, int(c[0]) - r), max(0, int(c[1]) - r)
        x1, y1 = min(W, int(c[0]) + r + 1), min(H, int(c[1]) + r + 1)
        if x1 <= x0 or y1 <= y0:
            return crop_and_resize(fr, c, s, 256, enlarge_ratio=1.0)[0]
        roi = fr[y0:y1, x0:x1]
        roi = cv2.GaussianBlur(roi, (5, 5), (d - 1) / 2) if d > 1.1 else roi
        return crop_and_resize(np.ascontiguousarray(roi), c - np.array([x0, y0], np.float32), s, 256, enlarge_ratio=1.0)[0]

    for vid in [v for v in a.ids.split(",") if v]:
        f_out = out / "3_gvhmr" / vid / "gvhmr_fast.npz"
        if f_out.exists():
            print(f"[dual] {vid}: exists, skipped", flush=True)
            continue
        if guard.skip(vid):
            continue
        try:
            video = Path(a.video_dir) / f"{vid}.mp4"
            bj_path = boxes_root / vid / "boxes.json"
            bj = json.loads(bj_path.read_text())
            if bj.get("no_person") or "padded_112" not in bj:
                print(f"[dual] {vid}: no person locked, skipped", flush=True)
                continue
            W, H, fps = int(bj["width"]), int(bj["height"]), float(bj["fps"])
            n = int(bj["n_frames"])
            present = np.asarray(bj["present"], bool)[:n]
            boxes = np.asarray(bj["padded_112"], np.float32)
            cuts = {int(c[0]) for c in bj.get("shot_cuts", [])}
            seg_of = -np.ones(n, np.int32)
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
            torch.cuda.reset_peak_memory_stats()
            t_all = time.perf_counter()
            tim = {"decode_crop": 0.0, "crop_full": 0.0, "vo": 0.0}
            with ProcessVramMonitor(baseline_mb=base) as mon:
                t0 = time.perf_counter()
                bxy = torch.zeros(n, 3)
                if use.any():
                    bxy[torch.from_numpy(use)] = get_bbx_xys_from_xyxy(torch.from_numpy(boxes[use]), base_enlarge=1.2).float()
                m = int(use.sum())
                crops = {v: np.zeros((m, 256, 256, 3), np.uint8) for v in variants}
                row = -np.ones(n, np.int64)
                row[use] = np.arange(m)
                vo_need = set()
                for (s_, e_) in segs:
                    vo_need |= set(range(s_, e_, VO_STEP)) | {e_ - 1}
                vo_frames = {}
                ph, w2, h2 = det._open_ffmpeg(video, W, H)
                pf = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-an", "-vsync", "0", "-f", "rawvideo",
                                       "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * 2) if do_full else None
                fbh, fbf = w2 * h2 * 3, W * H * 3
                k = 0
                while k < n:
                    bh = ph.stdout.read(fbh)
                    bf = pf.stdout.read(fbf) if do_full else b""
                    if len(bh) < fbh or (do_full and len(bf) < fbf):
                        break
                    if use[k] or k in vo_need:
                        frh = np.frombuffer(bh, np.uint8).reshape(h2, w2, 3)
                        if use[k]:
                            c, s_ = bxy[k, :2].numpy(), float(bxy[k, 2])
                            crops["half"][row[k]] = crop_half(frh, c, s_)
                            if do_full:
                                t1 = time.perf_counter()
                                crops["full"][row[k]] = crop_full(np.frombuffer(bf, np.uint8).reshape(H, W, 3), c, s_)
                                tim["crop_full"] += time.perf_counter() - t1
                        if k in vo_need:
                            vo_frames[k] = frh.copy()
                    k += 1
                for p in [x for x in (ph, pf) if x is not None]:
                    p.stdout.close()
                    p.terminate()
                    p.wait()
                if k < n:
                    print(f"[dual] {vid}: WARNING decoded {k} of {n} frames", flush=True)
                tim["decode_crop"] = time.perf_counter() - t0

                K = estimate_K(W, H).float()
                res = {}
                for v in variants:
                    o = {pre + f: np.full((n, d_), np.nan, np.float32) for pre in ("smpl_world_", "smpl_cam_")
                         for f, d_ in (("global_orient", 3), ("body_pose", 63), ("betas", 10), ("transl", 3))}
                    o.update(joints24_world=np.full((n, 24, 3), np.nan, np.float32), joints24_cam=np.full((n, 24, 3), np.nan, np.float32),
                             joints17_cam=np.full((n, 17, 3), np.nan, np.float32), kp2d=np.full((n, 17, 3), np.nan, np.float32))
                    res[v] = o
                    tim.update({f"{v}_vitpose": 0.0, f"{v}_hmr2": 0.0, f"{v}_gvhmr": 0.0, f"{v}_joints": 0.0})
                T_w2c = np.full((n, 4, 4), np.nan, np.float32)
                vo_failed = np.zeros(len(segs), bool)
                vo_errors, vo_frac = [], []
                for si, (s_, e_) in enumerate(segs):
                    L = e_ - s_
                    seg_bxy = bxy[s_:e_]
                    t0 = time.perf_counter()
                    use_static = False
                    try:
                        idxs = np.array(sorted(set(range(0, L, VO_STEP)) | {L - 1}))
                        T, fail = det._vo(np.stack([vo_frames[s_ + int(i)] for i in idxs]), idxs)
                        vo_frac.append(round(fail, 3))
                        if fail > 0.5:
                            raise RuntimeError(f"{fail:.0%} of frame pairs failed")
                    except Exception as exc:
                        vo_errors.append(f"seg{si}: {type(exc).__name__}: {exc}"[:200])
                        use_static = True
                    if use_static:
                        T = np.tile(np.eye(4, dtype=np.float32), (L, 1, 1))
                        vo_failed[si] = True
                    tim["vo"] += time.perf_counter() - t0
                    T_w2c[s_:e_] = T
                    angvel = compute_cam_angvel(torch.from_numpy(T[:, :3, :3]).float())
                    for v in variants:
                        o = res[v]
                        sc = crops[v][row[s_:e_]]
                        torch.cuda.synchronize(); t1 = time.perf_counter()
                        kp2d = det._vitpose(sc, seg_bxy)
                        torch.cuda.synchronize(); t2 = time.perf_counter()
                        feats = det._hmr2(sc)
                        torch.cuda.synchronize(); t3 = time.perf_counter()
                        data = {"length": torch.tensor(L), "bbx_xys": seg_bxy, "kp2d": kp2d, "K_fullimg": K.repeat(L, 1, 1),
                                "cam_angvel": angvel, "f_imgseq": feats}
                        pred = det.model.predict(data, static_cam=use_static)
                        torch.cuda.synchronize(); t4 = time.perf_counter()
                        pg = {kk: vv.detach() for kk, vv in pred["smpl_params_global"].items()}
                        pc = {kk: vv.detach() for kk, vv in pred["smpl_params_incam"].items()}
                        jw, _ = det._joints(pg)
                        jc, j17 = det._joints(pc, want17=True)
                        o["joints24_world"][s_:e_], o["joints24_cam"][s_:e_], o["joints17_cam"][s_:e_] = jw, jc, j17
                        o["kp2d"][s_:e_] = kp2d.numpy()
                        for kk in ("global_orient", "body_pose", "betas", "transl"):
                            o["smpl_world_" + kk][s_:e_] = pg[kk].cpu().numpy()
                            o["smpl_cam_" + kk][s_:e_] = pc[kk].cpu().numpy()
                        t5 = time.perf_counter()
                        for nm, dt in (("vitpose", t2 - t1), ("hmr2", t3 - t2), ("gvhmr", t4 - t3), ("joints", t5 - t4)):
                            tim[f"{v}_{nm}"] += dt
                        del pred, pg, pc
            total = time.perf_counter() - t_all
            common = dict(present=use, segment_id=seg_of.astype(np.int16),
                          segments=np.array([[s_, e_, int(vo_failed[i])] for i, (s_, e_) in enumerate(segs)], np.int32).reshape(-1, 3),
                          cam_T_w2c=T_w2c, cam_vo_failed=np.array([bool(vo_failed[s]) if s >= 0 else False for s in seg_of]),
                          K=K.numpy(), fps=np.float32(fps), width=np.int32(W), height=np.int32(H))
            for v, dst, dst2 in (("half", out / "3_gvhmr" / vid / "gvhmr_fast.npz", out / "2_pose2d" / vid / "vitpose_h.npz"),
                                 ("full", out / "_exp3_resolution" / vid / "gvhmr_full.npz", out / "_exp3_resolution" / vid / "pose2d_full.npz")):
                if v not in variants:
                    continue
                o = dict(res[v])
                kp = o.pop("kp2d")
                meta = {"model": "gvhmr_fast", "crop_source": "0.5x frames (IMG_DS)" if v == "half" else "full-resolution frames",
                        "vo_source": "0.5x frames", "fp16": True, "flip_test": False, "box_key": "padded_112", "min_seg": MIN_SEG,
                        "n_frames": n, "n_run": m, "n_segments": len(segs), "n_vo_failed_segments": int(vo_failed.sum()),
                        "vo_errors": vo_errors[:10], "vo_pair_fail_frac": vo_frac, "video": str(video), "boxes": str(bj_path)}
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dst, meta_json=np.asarray(json.dumps(meta)), **common, **o)
                dst2.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dst2, kp2d=kp, present=use, fps=np.float32(fps), width=np.int32(W), height=np.int32(H),
                                    meta_json=np.asarray(json.dumps({"model": "ViTPose-H (GVHMR vitpose-h-multi-coco), fp16, no flip test",
                                                                     "crop_source": meta["crop_source"], "layout": "COCO-17 [x_px, y_px, conf]",
                                                                     "box_key": "padded_112", "video": str(video)})))
            st = {"id": vid, "frames": n, "video_sec": round(n / fps, 2), "n_run": m, "wall_sec": round(total, 1),
                  "wall_over_video": round(total / (n / fps), 3), "timing_sec": {k_: round(v_, 2) for k_, v_ in tim.items()},
                  "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "nvsmi_peak_gb": round(mon.peak_gb, 2),
                  "segments": len(segs), "vo_failed_segments": int(vo_failed.sum())}
            st_path = out / ("_exp3_resolution" if do_full else "3_gvhmr") / vid / "stats.json"
            st_path.parent.mkdir(parents=True, exist_ok=True)
            st_path.write_text(json.dumps(st, indent=1))
            print("[dual]", json.dumps(st), flush=True)

        except Exception as e:  # noqa: BLE001
            guard.failed(vid, e)
            continue
        guard.ok()


if __name__ == "__main__":
    main()
