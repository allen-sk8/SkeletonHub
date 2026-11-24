"""World-space view of FineFS v2 GVHMR skeletons around one element (CPU only, matplotlib).

  python scripts/finefs_v2_world_view.py --id 0 --element 1 [--pad 1.5] [--out outputs/finefs_v2_world_view]

Video with three panels: original frame with the skater box | SMPL 24 joints in GVHMR world coordinates (Y up, metres,
ground grid at the lowest foot height of the clip, pelvis path drawn) | pelvis and lowest-foot height over time.
The world frame is the one of the GVHMR segment that covers the element (segments never share a world frame).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("/home/allen/datasets/FineFS")
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}


def t2s(x):
    m, s = x.split("-")
    return int(m) * 60 + int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--element", type=int, required=True, help="element number in the FineFS annotation (element1 = 1)")
    ap.add_argument("--pad", type=float, default=1.0, help="seconds added before and after the annotated time")
    ap.add_argument("--out", default="outputs/finefs_v2_world_view")
    ap.add_argument("--travel", action="store_true", help="with --fix: horizontal path from scripts/finefs_ground_track.py "
                    "(image-based travel over the ice) instead of GVHMR's; fixed view over the whole path")
    ap.add_argument("--fix", action="store_true", help="apply scripts/finefs_jump_height.py (re-ground + flight parabola); "
                    "the plot then shows the original heights dashed")
    a = ap.parse_args()
    ann = json.loads((D / "annotation" / f"{a.id}.json").read_text())["executed_element"][f"element{a.element}"]
    g = np.load(D / "3_gvhmr" / a.id / "gvhmr_fast.npz")
    bj = json.loads((D / "1_tracking" / a.id / "boxes.json").read_text())
    fps = float(g["fps"])
    s, e = [t2s(x) for x in ann["time"].split(",")]
    f0, f1 = max(0, int((s - a.pad) * fps)), min(len(g["present"]), int((e + 1 + a.pad) * fps))
    seg = g["segment_id"][f0:f1]
    main_seg = np.bincount(seg[seg >= 0]).argmax()
    keep = np.where(seg == main_seg)[0]
    f0, f1 = f0 + keep[0], f0 + keep[-1] + 1                    # stay inside one segment (one world frame)
    J = g["joints24_world"][f0:f1].astype(float)
    flights = []
    if a.fix:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from finefs_jump_height import find_flights, fix_vertical
        full_ann = json.loads((D / "annotation" / f"{a.id}.json").read_text())["executed_element"]
        flights = find_flights(g, full_ann, fps)
        dy = fix_vertical(g, flights, fps)[f0:f1]
        J0 = J.copy()
        J[:, :, 1] += dy[:, None]
        if a.travel:
            from finefs_ground_track import ground_velocity
            bj_ = dict(bj); bj_["id"] = a.id
            vel, _ = ground_velocity(g, bj_, f0, f1, flights=flights)
            path = J[0, 0, [0, 2]] + np.cumsum(np.vstack([[0, 0], vel[1:]]), axis=0) / fps
            J[:, :, [0, 2]] += (path - J[:, 0, [0, 2]])[:, None, :]
            speed = np.linalg.norm(vel, axis=1)
        pel0, foot0 = J0[:, 0, 1], np.minimum(J0[:, 10, 1], J0[:, 11, 1])
        ground0 = np.percentile(foot0, 5)
    pel, foot = J[:, 0, 1], np.minimum(J[:, 10, 1], J[:, 11, 1])
    ground = 0.0 if a.fix else np.percentile(foot, 5)
    # Display: matplotlib axes (X, Y, Z) = world (x, -z, y). Plotting (x, z, y) would be a mirror image (left-handed).
    J[:, :, 2] *= -1
    # Look from the broadcast camera's side, so left/right on screen matches the video:
    # camera optical axis in world = third row of R_c<-w = R_cam_body @ R_world_body^T (mean over the clip).
    from scipy.spatial.transform import Rotation as Rot
    Rw = Rot.from_rotvec(g["smpl_world_global_orient"][f0:f1]).as_matrix()
    Rc = Rot.from_rotvec(g["smpl_cam_global_orient"][f0:f1]).as_matrix()
    fwd = (Rc @ np.transpose(Rw, (0, 2, 1)))[:, 2, :].mean(0)
    azim = float(np.degrees(np.arctan2(fwd[2], -fwd[0])))       # viewer behind the camera, in display coordinates
    lo_, hi_ = J[:, :, [0, 2]].reshape(-1, 2).min(0), J[:, :, [0, 2]].reshape(-1, 2).max(0)
    ctr = (lo_ + hi_) / 2
    hx, hz = [max(1.2, (hi_[k] - lo_[k]) / 2 + 0.6) for k in (0, 1)]   # rectangle around the path
    half = max(hx, hz)
    name = f"{a.id}_e{a.element}_{ann['element']}" + ("_fixed" if a.fix else "") + ("_travel" if a.travel else "")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"_{name}_frames"; tmp.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(D / "video" / f"{a.id}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    t_ax = (np.arange(f0, f1) - f0) / fps
    a0, a1 = (s * fps - f0) / fps, ((e + 1) * fps - f0) / fps
    fig = plt.figure(figsize=(19, 6.4), dpi=80)
    for i in range(f1 - f0):
        ok, fr = cap.read()
        fig.clf()
        ax0 = fig.add_axes([0.0, 0.05, 0.27, 0.9]); ax0.axis("off")
        if ok:
            b = bj["tight_clean"][f0 + i]
            if bj["present"][f0 + i] and b is not None:
                x1, y1, x2, y2 = map(int, b); cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            ax0.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        ax0.set_title(f"video {a.id}  {ann['element']}  frame {f0 + i}  t={(f0 + i) / fps:.2f}s", fontsize=10)
        ax = fig.add_axes([0.25, -0.08, 0.48, 1.16], projection="3d")
        j = J[i]
        for c, p in enumerate(PARENTS):
            if p >= 0:
                ax.plot([j[c, 0], j[p, 0]], [j[c, 2], j[p, 2]], [j[c, 1], j[p, 1]], color="tab:red" if c in LEFT else "tab:blue", lw=2.5)
        if a.travel:                                              # follow the skater; whole path drawn on the ice
            pth = J[:, 0, [0, 2]]                                  # fixed view over the whole path; travel is visible
            ax.plot(pth[:, 0], pth[:, 1], np.full(len(pth), ground), color="0.8", lw=1)
            ax.plot(pth[:i + 1, 0], pth[:i + 1, 1], np.full(i + 1, ground), color="k", lw=1.5)
        else:
            ax.plot(J[:i + 1, 0, 0], J[:i + 1, 0, 2], J[:i + 1, 0, 1] * 0 + ground, color="gray", lw=1)   # pelvis path on the ground
        step = 1.0 if half < 5 else 2.0                           # ground grid every 1 m (2 m on long paths)
        gx = np.arange(np.ceil((ctr[0] - hx) / step) * step, ctr[0] + hx + 1e-6, step)
        gz = np.arange(np.ceil((ctr[1] - hz) / step) * step, ctr[1] + hz + 1e-6, step)
        for x in gx: ax.plot([x, x], [ctr[1] - hz, ctr[1] + hz], [ground, ground], color="0.85", lw=0.6)
        for z in gz: ax.plot([ctr[0] - hx, ctr[0] + hx], [z, z], [ground, ground], color="0.85", lw=0.6)
        zt = 2.35                                                 # equal metres on all axes
        ax.set_xlim(ctr[0] - hx, ctr[0] + hx); ax.set_ylim(ctr[1] - hz, ctr[1] + hz); ax.set_zlim(ground - 0.05, ground - 0.05 + zt)
        ax.set_box_aspect((2 * hx, 2 * hz, zt), zoom=1.12 if a.travel else 1.0); ax.view_init(elev=25 if a.travel else 12, azim=azim)
        ax.set_xlabel("x (m)"); ax.set_ylabel("-z (m)"); ax.set_zlabel("up (m)" if not a.travel else ""); ax.set_zticks([0, 1, 2] if a.travel else ax.get_zticks())
        ax.set_title(("image-based travel, " if a.travel else "") + ("re-grounded + flight parabola" if a.fix else "GVHMR world (one segment, Y up)"),
                     fontsize=10)
        ax2 = fig.add_axes([0.75, 0.12, 0.2, 0.78])
        ax2.axvspan(a0, a1, color="0.92", label="annotated element time")
        for fa, fb, r, how, el in flights:
            if fb > f0 and fa < f1:
                ax2.axvspan((fa - f0) / fps, (fb - f0) / fps, color="tab:green", alpha=0.25,
                            label=f"flight ({'from rotation' if how == 'rotation' else 'typical time'})")
        if a.fix:
            ax2.plot(t_ax, (pel0 - ground0) * 100, "--", color="tab:blue", alpha=0.5, label="pelvis, GVHMR original")
            ax2.plot(t_ax, (foot0 - ground0) * 100, "--", color="tab:orange", alpha=0.5, label="foot, GVHMR original")
        ax2.plot(t_ax, (pel - ground) * 100, label="pelvis height" + (" (fixed)" if a.fix else ""), color="tab:blue")
        ax2.plot(t_ax, (foot - ground) * 100, label="lowest foot height" + (" (fixed)" if a.fix else ""), color="tab:orange")
        ax2.axvline(t_ax[i], color="k", lw=1)
        ax2.set_xlabel("seconds"); ax2.set_ylabel("cm above ground (5th pct. of lowest foot)")
        if a.travel:
            ax3 = ax2.twinx()
            ax3.plot(t_ax, speed, color="tab:purple", lw=1, label="speed over the ice")
            ax3.set_ylim(0, 12); ax3.set_ylabel("speed over the ice (m/s)", color="tab:purple")
            ax3.text(0.98, 0.97, f"speed {speed[i]:.1f} m/s", transform=ax3.transAxes, ha="right", va="top", color="tab:purple")
        hh, ll = ax2.get_legend_handles_labels(); keep_ = dict(zip(ll, hh))
        ax2.legend(keep_.values(), keep_.keys(), fontsize=7, loc="upper left"); ax2.grid(alpha=0.3)
        fig.savefig(tmp / f"{i:05d}.png")
    cap.release(); plt.close(fig)
    mp4 = out / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", str(mp4)], check=True)
    for p in tmp.glob("*.png"): p.unlink()
    tmp.rmdir()
    print(f"{mp4}  frames {f0}-{f1}  pelvis range {np.ptp(pel)*100:.0f} cm  lowest foot max {(foot.max()-ground)*100:.0f} cm")


if __name__ == "__main__":
    main()
