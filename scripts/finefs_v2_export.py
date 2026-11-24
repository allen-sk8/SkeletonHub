"""FineFS v2 test output: stage folders 1_tracking, 4_flags, 5_preview, info.txt in every stage folder, and format
checks. 2_pose2d and 3_gvhmr are written by scripts/finefs_v2_gvhmr_dual.py. CPU only (cv2 + ffmpeg), any env with cv2.

  python scripts/finefs_v2_export.py stage1 --ids 36,261 --boxes-from outputs/finefs_v2_test/_exp1_offline \
         --tracks-from outputs/finefs_full_test            # 1_tracking (before the GPU stage)
  python scripts/finefs_v2_export.py stage4 --ids ...      # 4_flags, 5_preview, info.txt, check.json (after the GPU stage)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "finefs_v2_test"
VIDEO_DIR = Path("/home/allen/datasets/FineFS/video")
CLOSEUP_H = 0.90
COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 1), (0, 2), (1, 3), (2, 4)]
PREVIEW_H = 480


def git_rev():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "preprocessors", "detectors", "scripts"],
                           capture_output=True, text=True)
        return r.stdout.strip() + (" (+ uncommitted changes in code dirs)" if d.stdout.strip() else "")
    except Exception:
        return "unknown"


def stage1(ids, boxes_from: Path, tracks_from: Path):
    for vid in ids:  # stage 0 = the original videos in VIDEO_DIR (no links or copies)
        d = OUT / "1_tracking" / vid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(boxes_from / vid / "boxes.json", d / "boxes.json")
        for f in ("tracks.txt", "detections_medium.npz"):
            src = tracks_from / vid / f
            shutil.copyfile(src.resolve(), d / f)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "finefs_cut_sheet.py"), "--id", vid, "--out", str(OUT / "1_tracking")],
                       check=True)
        print(f"[stage1] {vid} done", flush=True)


def flags(vid):
    bj = json.loads((OUT / "1_tracking" / vid / "boxes.json").read_text())
    n, H = int(bj["n_frames"]), int(bj["height"])
    present = np.asarray(bj["present"], bool)[:n]
    tb = np.asarray(bj.get("tight_clean", [[0, 0, 0, 0]] * n), np.float64)[:n]
    closeup = present & ((tb[:, 3] - tb[:, 1]) > CLOSEUP_H * H)
    d = OUT / "4_flags" / vid
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "flags.npz", closeup_face=closeup, replay=np.full(n, -1, np.int8), dissolve=np.full(n, -1, np.int8))
    s = {"n_frames": n, "closeup_face_frames": int(closeup.sum()), "closeup_face_ratio": round(float(closeup.mean()), 4),
         "replay": "not detected yet (all -1)", "dissolve": "not detected yet (all -1)"}
    (d / "flags.json").write_text(json.dumps(s, indent=1))
    return closeup


def preview(vid, closeup):
    bj = json.loads((OUT / "1_tracking" / vid / "boxes.json").read_text())
    n, W, H = int(bj["n_frames"]), int(bj["width"]), int(bj["height"])
    present = np.asarray(bj["present"], bool)[:n]
    pad = bj.get("padded_112")
    kp = np.load(OUT / "2_pose2d" / vid / "vitpose_h.npz")["kp2d"]
    g = np.load(OUT / "3_gvhmr" / vid / "gvhmr_fast.npz")
    seg_id = g["segment_id"]
    vo_failed = g["cam_vo_failed"]
    fps = float(bj["fps"])
    sc = PREVIEW_H / H
    size = (int(round(W * sc / 2)) * 2, PREVIEW_H)
    d = OUT / "5_preview"
    d.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                          "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                          str(d / f"{vid}.mp4")], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{vid}.mp4"))
    t = 0
    while t < n:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
        if present[t] and pad:
            b = np.asarray(pad[t]) * sc
            col = (0, 165, 255) if closeup[t] else (60, 245, 70)
            cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        if np.isfinite(kp[t, 0, 0]):
            k = kp[t].copy()
            k[:, :2] *= sc
            for i, j in COCO_EDGES:
                if k[i, 2] >= 0.3 and k[j, 2] >= 0.3:
                    cv2.line(fr, (int(k[i, 0]), int(k[i, 1])), (int(k[j, 0]), int(k[j, 1])), (255, 200, 0), 2)
            for q in k:
                if q[2] >= 0.3:
                    cv2.circle(fr, (int(q[0]), int(q[1])), 2, (0, 0, 255), -1)
        txt = f"{vid} f{t} {t / fps:.1f}s"
        if not present[t]:
            txt += "  NO PERSON"
        elif seg_id[t] < 0:
            txt += "  (segment <10 f, no 3D)"
        else:
            txt += f"  seg {int(seg_id[t])}" + ("  static cam" if vo_failed[t] else "")
        if closeup[t]:
            txt += "  CLOSE-UP"
        cv2.putText(fr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(fr, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        p.stdin.write(fr.tobytes())
        t += 1
    cap.release()
    p.stdin.close()
    p.wait()


def ffprobe_frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets,width,height", "-of", "json", str(path)], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return int(s["nb_read_packets"]), int(s["width"]), int(s["height"])


def check(vid):
    c = {"id": vid}
    nv, W, H = ffprobe_frames(VIDEO_DIR / f"{vid}.mp4")
    bj = json.loads((OUT / "1_tracking" / vid / "boxes.json").read_text())
    g = np.load(OUT / "3_gvhmr" / vid / "gvhmr_fast.npz")
    k = np.load(OUT / "2_pose2d" / vid / "vitpose_h.npz")
    f = np.load(OUT / "4_flags" / vid / "flags.npz")
    det = np.load(OUT / "1_tracking" / vid / "detections_medium.npz")
    n = int(bj["n_frames"])
    lens = {"video_packets": nv, "boxes.n_frames": n, "boxes.present": len(bj["present"]), "boxes.padded_112": len(bj["padded_112"]),
            "gvhmr.present": len(g["present"]), "gvhmr.joints24_world": len(g["joints24_world"]), "pose2d.kp2d": len(k["kp2d"]),
            "flags.closeup_face": len(f["closeup_face"]), "detections.num_frames": int(det["num_frames"])}
    c["lengths"] = lens
    c["lengths_match"] = len(set(lens.values())) == 1
    run = g["present"]
    c["kp2d_saved_frames"] = int(np.isfinite(k["kp2d"][:, 0, 0]).sum())
    c["kp2d_matches_3d_frames"] = bool((np.isfinite(k["kp2d"][:, 0, 0]) == run).all())
    c["joints_finite_where_present"] = bool(np.isfinite(g["joints24_world"][run]).all())
    np_, pw, ph = ffprobe_frames(OUT / "5_preview" / f"{vid}.mp4")
    c["preview"] = {"frames": np_, "size": f"{pw}x{ph}", "frames_match": np_ == n}
    cap = cv2.VideoCapture(str(OUT / "5_preview" / f"{vid}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    c["preview"]["decodes_mid_frame"] = bool(cap.read()[0])
    cap.release()
    mb = {}
    for st in ("1_tracking", "2_pose2d", "3_gvhmr", "4_flags"):
        mb[st] = round(sum(p.stat().st_size for p in (OUT / st / vid).rglob("*") if p.is_file()) / 1e6, 2)
    mb["1_tracking_without_detections"] = round(mb["1_tracking"] - (OUT / "1_tracking" / vid / "detections_medium.npz").stat().st_size / 1e6, 2)
    mb["5_preview"] = round((OUT / "5_preview" / f"{vid}.mp4").stat().st_size / 1e6, 2)
    c["mb"] = mb
    c["video_sec"] = round(n / float(bj["fps"]), 1)
    return c


INFO = {
    "": """FineFS whole-match videos, v2 stage output ({nvid} videos so far, info written {date}). Stage folders, one sub-folder or file per video id
(= FineFS whole-match video number, /home/allen/datasets/FineFS/video/<id>.mp4).
Code version: SkeletonHub commit {rev}. Code: SkeletonHub/scripts/run_finefs_v2_batch.sh. Test report: SkeletonHub/docs/FINEFS_V2_TEST_2026-11-23.md
Frame index t is the same in every stage (0 .. n_frames-1 of the original video, 25 fps). Absent frames: NaN / False / -1.
COORDINATES: every 3D segment (3_gvhmr "segments") has its OWN world frame (GVHMR, Y up, metres); segments are split at camera
cuts and at frames with no person, and world coordinates of different segments can NOT be joined. Camera translation from
SimpleVO is up to an unknown scale (only its rotation is used by GVHMR). No 3D ground truth exists for these videos.
VIDEOS: stage 0 is the original FineFS folder video/ (<id>.mp4); it is not duplicated. Numbering starts at 1 on purpose.
Stages: 1_tracking 2_pose2d 3_gvhmr 4_flags 5_preview (each has its own info.txt). 4_flags/<id>/check.json = per-video
format checks (lengths of every stage agree with the video, preview decodes).
""",
    "1_tracking": """Person tracking and lock-on of the skater. Per video <id>/:
  boxes.json   width, height, n_frames, fps; present[t] (is the skater in frame t); shot_cuts [[first frame of new shot, diff]];
               tight_clean[t] / padded_112[t] / padded_125[t] boxes [x1,y1,x2,y2] in original pixels (padded_112 = box used by
               GVHMR; meaningless where present is false); raw_D / raw_D_ids (box and track id before repair/interpolation);
               target = "{target}"; lock = audit trail: seed track, every attached tracklet with its appearance similarity
               ("sim", OSNet mean cosine to the skater gallery; accepted >= 0.6, or >= 0.55 for a big central person), and the
               best rejected ones.
  tracks.txt   all BoT-SORT tracks, MOT rows frame,id,x1,y1,x2,y2,conf  (tracker: {tracker})
  detections_medium.npz  RF-DETR Medium person detections of every frame (frame_idx, xyxy, confidence, num_frames); the cache
               that tracking and lock-on read; can be deleted after tracking if disk space matters.
  cut_sheet.png  contact sheet: frames just before / after camera cuts and person present/absent changes, green = skater box.
Settings: RF-DETR Medium (conf >= 0.2) -> BoT-SORT ({tracker}) -> shot cuts from thumbnail differences -> {lock_desc}
""",
    "2_pose2d": """2D keypoints from ViTPose-H (GVHMR's vitpose-h-multi-coco weights), the exact 2D input of the 3D stage. Per video
<id>/vitpose_h.npz: kp2d (T,17,3) float32 [x_px, y_px, confidence] in ORIGINAL video pixels, COCO-17 order
(nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee,
l_ankle, r_ankle); NaN where no 3D was run. present (T,) bool, fps, width, height, meta_json.
Settings: fp16, no flip test, crop 256x256 around padded_112 box (GVHMR enlarge 1.2), crop cut from {crop_src}.
""",
    "3_gvhmr": """3D body from gvhmr_fast (GVHMR, fp16 ViTPose-H and HMR2 backbones). Per video <id>/gvhmr_fast.npz, length T arrays:
  present (T,) frames that were run; segment_id (T,) int16 (-1 = not run); segments (S,3) [start, end_exclusive, vo_failed]
  smpl_world_{{global_orient (T,3), body_pose (T,63), betas (T,10), transl (T,3)}}  GVHMR smpl_params_global, world Y up, metres
  smpl_cam_{{global_orient, body_pose, betas, transl}}                              GVHMR smpl_params_incam, OpenCV camera
  joints24_world, joints24_cam (T,24,3)  SMPL 24 joints; joints17_cam (T,17,3) H36M-17 (camera space)
  cam_T_w2c (T,4,4) SimpleVO camera pose (rotation used by GVHMR; translation only up to scale; identity if vo_failed)
  cam_vo_failed (T,) static camera assumed because SimpleVO failed on that segment
  K (3,3) intrinsics GVHMR assumes (estimate_K: focal from image size, principal point at centre), fps, width, height
  meta_json: settings and timings.
Segments: runs of present frames not crossing a camera cut, at least 10 frames; each segment = one GVHMR sequence with its own
world frame (do not join segments). Crops cut from {crop_src}; SimpleVO on 0.5x frames, failure-tolerant (static camera when
> 50% of frame pairs fail). Read with SkeletonHub/hmr_zoo/fast_io.py (load_fast, to_mstcn_features).
""",
    "4_flags": """Per-frame quality flags. Per video <id>/flags.npz + flags.json (counts):
  closeup_face (T,) bool  skater box (tight_clean) taller than {closeup:.0%} of the frame height: face / upper-body close-ups at the
                          start and end of a programme; the 3D for these frames is not meaningful.
  replay (T,) int8        slow-motion replay. NOT DETECTED YET: all -1 (= unknown). 1 / 0 once a detector exists.
  dissolve (T,) int8      cross-dissolve between two shots (box may lock onto the fading image). NOT DETECTED YET: all -1.
""",
    "5_preview": """Low-resolution check videos <id>.mp4 ({ph}p, H.264 crf 28): box (green; orange = close-up flag), ViTPose 2D skeleton
(points with confidence >= 0.3), and a text line: frame, time, NO PERSON / segment id / static cam / CLOSE-UP.
""",
}


def write_info(target, tracker, lock_desc, crop_src):
    import datetime
    rev = git_rev()
    nvid = sum(1 for d in (OUT / "3_gvhmr").iterdir() if (d / "gvhmr_fast.npz").exists())
    for st, txt in INFO.items():
        p = OUT / st / "info.txt" if st else OUT / "stages_info.txt"  # root file named so it is clear next to other dataset folders
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt.format(date=datetime.date.today().isoformat(), rev=rev, target=target, tracker=tracker, lock_desc=lock_desc,
                                crop_src=crop_src, closeup=CLOSEUP_H, ph=PREVIEW_H, nvid=nvid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage4"])
    ap.add_argument("--ids", default="36,261,819,739,980,652,187,513,521,456")
    ap.add_argument("--boxes-from")
    ap.add_argument("--tracks-from")
    ap.add_argument("--root", default=None, help="output root (default outputs/finefs_v2_test)")
    ap.add_argument("--no-summary", action="store_true", help="stage4: only flags/preview/check per video, no check.json/info.txt")
    ap.add_argument("--summary-dir", default=None, help="stage4: where the batch summary check.json goes (default: --root); "
                    "set it to a SkeletonHub work folder so the dataset folder only holds stage folders")
    ap.add_argument("--fail-dir", default=None, help="record per-video failures here and go on (see finefs_failguard.py)")
    ap.add_argument("--n-total", type=int, default=1167)
    ap.add_argument("--hours-total", type=float, default=65.3)
    a = ap.parse_args()
    global OUT
    if a.root:
        OUT = Path(a.root).resolve()
    ids = [v for v in a.ids.split(",") if v]
    if a.stage == "stage1":
        stage1(ids, Path(a.boxes_from), Path(a.tracks_from))
        return
    checks = []
    sys.path.insert(0, str(ROOT / "scripts"))
    from finefs_failguard import FailGuard
    guard = FailGuard(a.fail_dir, "export")
    for vid in ids:
        if not (OUT / "3_gvhmr" / vid / "gvhmr_fast.npz").exists():
            print(f"[export] {vid}: no 3D result, skipped", flush=True)
            continue
        if guard.skip(vid):
            continue
        try:
            cl = flags(vid)
            if not (OUT / "5_preview" / f"{vid}.mp4").exists():
                preview(vid, cl)
            c = check(vid)
        except Exception as e:  # noqa: BLE001
            guard.failed(vid, e)
            continue
        guard.ok()
        (OUT / "4_flags" / vid / "check.json").write_text(json.dumps(c, indent=1))
        checks.append(c)
        print("[check]", json.dumps(c), flush=True)
    if a.no_summary or not checks:
        return
    bj = json.loads((OUT / "1_tracking" / ids[0] / "boxes.json").read_text())
    write_info(bj.get("target"), bj.get("tracker"),
               "offline lock-on per shot (preprocessors/track_person_offline.py)" if bj.get("target") == "lock_offline"
               else "online lock-on (track_person.py --target lock)", "0.5x frames (GVHMR IMG_DS, current default)")
    tot_sec = sum(c["video_sec"] for c in checks)
    keys = list(checks[0]["mb"])
    per_hour = {k: sum(c["mb"][k] for c in checks) / (tot_sec / 3600) for k in keys}
    summary = {"all_lengths_match": all(c["lengths_match"] for c in checks),
               "all_kp2d_match": all(c["kp2d_matches_3d_frames"] for c in checks),
               "all_previews_ok": all(c["preview"]["frames_match"] and c["preview"]["decodes_mid_frame"] for c in checks),
               "mb_per_video_mean": {k: round(sum(c["mb"][k] for c in checks) / len(checks), 2) for k in keys},
               "gb_full_batch_by_count": {k: round(sum(c["mb"][k] for c in checks) / len(checks) * a.n_total / 1000, 1) for k in keys},
               "gb_full_batch_by_hours": {k: round(per_hour[k] * a.hours_total / 1000, 1) for k in keys},
               "videos": checks}
    sd = Path(a.summary_dir) if a.summary_dir else OUT
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "videos"}, indent=1))


if __name__ == "__main__":
    main()
