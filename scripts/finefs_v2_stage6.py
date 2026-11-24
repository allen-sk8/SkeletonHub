"""FineFS v2 stage 6: skater on the ice (CPU only). Spec: docs/FINEFS_V2_STAGE6_SPEC.md.

Per video writes <root>/6_world/<id>/world.npz and jumps.json (written as .part, renamed when complete):
  offset_rotation, offset_image (T,3)  add to 3_gvhmr joints24_world / smpl_world_transl (metres, GVHMR world of each
                                       segment, Y up; after correction the ice is y = 0); NaN where there is no 3D
  ground_dy (T,)                       the re-grounding part alone (no jumps)
  travel_xz (T,2)                      the horizontal part alone (path minus GVHMR's own pelvis x, z)
  speed (T,), speed_capped (T,)        speed over the ice (m/s)
  homography (T,3,3), homography_inliers (T,)   background motion t-1 -> t, full-resolution pixels (NaN: failed)
  meta_json

  python scripts/finefs_v2_stage6.py --ids 0,8                     # given videos
  python scripts/finefs_v2_stage6.py --watch --work outputs/finefs_v2_full --worker 1
        # every video whose stages 1-5 are done and stage 6 is not; several workers share the work through lock
        # folders in <work>/_s6lock; ends when the GPU batch is ALL_DONE (or stopped for good) and nothing is left;
        # touch <work>/STOP6 to end the workers after their current video.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # CPU only: an imported library otherwise opens a context on the default card
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finefs_ground_track import ELEV_DEG, MAX_SPEED, background_homographies, ground_velocity  # noqa: E402
from finefs_jump_height import G, find_flights, fix_vertical  # noqa: E402
from finefs_jump_image import LIFT_ON, measure_jump  # noqa: E402

ROOT_DS = Path("/home/allen/datasets/FineFS")
REPO = Path(__file__).resolve().parents[1]


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def process(vid, R: Path):
    g = np.load(R / "3_gvhmr" / vid / "gvhmr_fast.npz")
    bj = json.loads((R / "1_tracking" / vid / "boxes.json").read_text()); bj["id"] = vid
    kp = np.load(R / "2_pose2d" / vid / "vitpose_h.npz")["kp2d"]
    ann = json.loads((R / "annotation" / f"{vid}.json").read_text())["executed_element"]
    fps = float(g["fps"])
    n = len(g["present"])
    seg = g["segment_id"]
    video = R / "video" / f"{vid}.mp4"
    t0 = time.time()
    boxes = [bj["tight_clean"][t] if bj["present"][t] else None for t in range(len(bj["present"]))]
    H, inl = background_homographies(video, boxes, 0, n)
    hom = (H, inl)
    t_hom = time.time() - t0

    # jumps: rotation method, then image method on the same jump parts
    fl_rot = find_flights(g, ann, fps)
    fl_img, jumps = [], []
    for k, (a, b, revs, how, el) in enumerate(fl_rot):
        e = next((e for e in ann.values() if e["element"] == el), {})
        m = measure_jump(g, bj, kp, (a + b) // 2, fps, video, hom=hom)
        rec = {"part": k, "element": el, "revolutions": revs, "goe": e.get("goe"), "annotated_time": e.get("time"),
               "rotation": {"flight_frames": [int(a), int(b)], "flight_s": round((b - a) / fps, 3),
                            "how": how, "height_m": round(G * ((b - a) / fps) ** 2 / 8, 3)},
               "image": {"ok": bool(m["ok"]), "why": m.get("why")}}
        if "T_img" in m:
            rec["image"].update(takeoff_frame=round(float(m["t_takeoff"]), 2), landing_frame=round(float(m["t_landing"]), 2),
                                flight_s=round(float(m["T_img"]), 3), apex_height_m=round(float(m["h_img"]), 3),
                                height_from_flight_m=round(float(m["h_from_T"]), 3),
                                height_and_flight_agree=bool(abs(m["h_img"] - m["h_from_T"]) < 0.15))
        if m["ok"]:
            fl_img.append((int(round(m["t_takeoff"])), int(round(m["t_landing"])) + 1, revs, "image", el, float(m["h_img"])))
            rec["image_variant_uses"] = "image"
        else:
            fl_img.append((a, b, revs, how, el))
            rec["image_variant_uses"] = "rotation (image measurement failed)"
        jumps.append(rec)

    dy_rot = fix_vertical(g, fl_rot, fps)
    dy_img = fix_vertical(g, fl_img, fps)
    dy_ground = fix_vertical(g, [], fps)

    # travel over the ice, per segment
    J = g["joints24_world"]
    travel = np.full((n, 2), np.nan, np.float32)
    speed = np.full(n, np.nan, np.float32)
    capped = np.zeros(n, bool)
    for s_ in np.unique(seg[seg >= 0]):
        idx = np.where(seg == s_)[0]
        a, b = int(idx[0]), int(idx[-1]) + 1
        if b - a < 3:
            continue
        v, _ = ground_velocity(g, bj, a, b, flights=fl_rot, hom=hom)
        sp = np.linalg.norm(v, axis=1)
        path = J[a, 0, [0, 2]] + np.cumsum(np.vstack([[0, 0], v[1:]]), axis=0) / fps
        travel[a:b] = path - J[a:b, 0][:, [0, 2]]
        speed[a:b] = sp
        capped[a:b] = sp >= MAX_SPEED - 1e-6
    has3d = seg >= 0

    def offset(dy):
        o = np.full((n, 3), np.nan, np.float32)
        o[has3d, 0] = travel[has3d, 0]
        o[has3d, 1] = dy[has3d]
        o[has3d, 2] = travel[has3d, 1]
        return o

    meta = {"stage": "6_world", "code": f"SkeletonHub {git_rev()} scripts/finefs_v2_stage6.py",
            "spec": "SkeletonHub/docs/FINEFS_V2_STAGE6_SPEC.md", "camera_elevation_deg": ELEV_DEG,
            "max_speed_mps": MAX_SPEED, "lift_on_m": LIFT_ON, "fps": fps, "n_frames": n,
            "n_jump_parts": len(jumps), "n_image_ok": sum(j["image"]["ok"] for j in jumps),
            "n_rotation_measured": sum(j["rotation"]["how"] == "rotation" for j in jumps),
            "homography_failed_frames": int((~np.isfinite(H[:, 0, 0])).sum()),
            "seconds": {"homography": round(t_hom, 1), "total": round(time.time() - t0, 1)},
            "use": "corrected joints = 3_gvhmr joints24_world + offset_<variant>[:, None, :]; corrected SMPL transl = "
                   "smpl_world_transl + offset_<variant>. Segments keep separate world frames."}
    d = R / "6_world" / vid
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "jumps.json.part", "w") as fh:
        json.dump({"id": vid, "jumps": jumps}, fh, indent=1)
    with open(d / "world.npz.part", "wb") as fh:
        np.savez_compressed(fh, offset_rotation=offset(dy_rot), offset_image=offset(dy_img),
                            ground_dy=np.where(has3d, dy_ground, np.nan).astype(np.float32), travel_xz=travel,
                            speed=speed, speed_capped=capped, homography=H.astype(np.float32),
                            homography_inliers=inl.astype(np.float32), meta_json=np.asarray(json.dumps(meta)))
    os.replace(d / "jumps.json.part", d / "jumps.json")
    os.replace(d / "world.npz.part", d / "world.npz")           # world.npz = the "done" marker
    return meta


INFO = """Stage 6: the skater on the ice (CPU only; spec SkeletonHub/docs/FINEFS_V2_STAGE6_SPEC.md).
GVHMR's body pose is kept; only the whole-body position changes, so each variant is a per-frame offset in metres:
  corrected joints = 3_gvhmr/<id>/gvhmr_fast.npz joints24_world + offset_<variant>[:, None, :]
  corrected SMPL transl = smpl_world_transl + offset_<variant>
Per video <id>/world.npz:
  offset_rotation (T,3)  feet put on the ice; jumps: flight time from body rotation (or the typical flight time for the
                         number of revolutions when the rotation is unclear), height g*T^2/8 as a parabola of the pelvis
  offset_image (T,3)     same, but flight and height measured in the image (ankle lift above the ice); jumps where the
                         image measurement failed use the rotation flight (marked in jumps.json)
  both: horizontal travel over the ice measured in the image (camera motion from the background removed, pixels to
        metres with GVHMR's depth, camera assumed to look down 20 degrees, speed capped at 12 m/s)
  ground_dy (T,), travel_xz (T,2)  the two parts alone; speed (T,) m/s, speed_capped (T,)
  homography (T,3,3)     background motion from frame t-1 to t (full-resolution pixels; NaN = failed), homography_inliers
  meta_json              settings, code version
<id>/jumps.json: every jump part from the FineFS annotation (a combination has one entry per jump): element,
  revolutions, GOE, annotated time; rotation method (flight frames, measured or typical, height); image method (ok or
  why not, take-off / landing frame (sub-frame), flight time, apex height, height from flight time, whether they agree
  within 15 cm); which flight the image variant used.
Coordinates: GVHMR world of each segment (Y up, metres; the ice is y = 0 after correction). Segments are still separate
world frames. NaN where 3_gvhmr has no 3D. Jumps without FineFS annotation are not handled.
"""


def todo_ids(R: Path, ids):
    out = []
    for v in ids:
        if (R / "6_world" / v / "world.npz").exists():
            continue
        if (R / "3_gvhmr" / v / "gvhmr_fast.npz").exists() and (R / "5_preview" / f"{v}.mp4").exists() \
                and (R / "4_flags" / v / "flags.npz").exists():
            out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None)
    ap.add_argument("--root", default=str(ROOT_DS))
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--work", default="outputs/finefs_v2_full", help="GPU batch work folder (all_ids.txt, ALL_DONE, STOP)")
    ap.add_argument("--worker", default="0")
    ap.add_argument("--preview-every", type=int, default=20, help="watch mode: one preview jump for every Nth video id")
    ap.add_argument("--preview-dir", default="outputs/finefs_v2_stage6_previews")
    a = ap.parse_args()
    R = Path(a.root)
    (R / "6_world").mkdir(exist_ok=True)
    (R / "6_world" / "info.txt").write_text(INFO)
    if not a.watch:
        for v in a.ids.split(","):
            m = process(v, R)
            print(f"[stage6] {v}: {m['n_jump_parts']} jump parts, image ok {m['n_image_ok']}, rotation measured "
                  f"{m['n_rotation_measured']}, {m['seconds']['total']} s", flush=True)
        return
    W = Path(a.work)
    lock, failed = W / "_s6lock", W / "_failed6"
    lock.mkdir(parents=True, exist_ok=True); failed.mkdir(parents=True, exist_ok=True)
    all_ids = (W / "all_ids.txt").read_text().split()
    prev_set = set(all_ids[::a.preview_every])
    while True:
        if (W / "STOP6").exists():
            print("[stage6] STOP6 found, ending", flush=True); return
        cand = [v for v in todo_ids(R, all_ids) if not (failed / f"{v}.txt").exists()]
        took = None
        for v in cand:
            try:
                (lock / v).mkdir()
                took = v; break
            except FileExistsError:
                continue
        if took is None:
            sup = (W / "supervisor.last").read_text() if (W / "supervisor.last").exists() else ""
            gpu_over = (W / "ALL_DONE").exists() or sup.startswith("stopped")   # STOP alone can be a card switch
            if gpu_over and not cand:
                print("[stage6] nothing left and the GPU batch is not running, ending", flush=True); return
            time.sleep(300); continue
        try:
            m = process(took, R)
            print(f"[stage6 w{a.worker}] {took}: {m['n_jump_parts']} jump parts, image ok {m['n_image_ok']}, "
                  f"{m['seconds']['total']} s", flush=True)
            if took in prev_set:
                make_preview(took, R, Path(a.preview_dir))
        except Exception as e:  # noqa: BLE001
            (failed / f"{took}.txt").write_text(f"{time.strftime('%F %T')} {type(e).__name__}: {e}\n{traceback.format_exc()}")
            print(f"[stage6 w{a.worker}] {took}: FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            try:
                (lock / took).rmdir()
            except OSError:
                pass


def make_preview(vid, R: Path, out: Path):
    """One jump of this video as a three-panel world view (image height variant): the first jump whose image
    measurement worked, else the first jump."""
    js = json.loads((R / "6_world" / vid / "jumps.json").read_text())["jumps"]
    if not js:
        return
    pick = next((j for j in js if j["image"]["ok"]), js[0])
    ann = json.loads((R / "annotation" / f"{vid}.json").read_text())["executed_element"]
    el = next((k for k, e in ann.items() if e["element"] == pick["element"]), None)
    if el is None:
        return
    py = sys.executable
    r = subprocess.run([py, str(REPO / "scripts" / "finefs_v2_world_view.py"), "--id", vid, "--element", el[len("element"):],
                        "--fix", "--travel", "--height", "image", "--out", str(out)], capture_output=True, text=True, cwd=str(REPO))
    print(f"[stage6] preview {vid}: {r.stdout.strip().splitlines()[-1] if r.returncode == 0 else 'FAILED ' + r.stderr[-200:]}", flush=True)


if __name__ == "__main__":
    main()
