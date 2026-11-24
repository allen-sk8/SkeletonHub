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
    pel, foot = J[:, 0, 1], np.minimum(J[:, 10, 1], J[:, 11, 1])
    ground = np.percentile(foot, 5)
    ctr = J[:, 0, [0, 2]].mean(0)
    half = max(1.2, np.abs(J[:, :, [0, 2]] - ctr).max() + 0.3)
    name = f"{a.id}_e{a.element}_{ann['element']}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"_{name}_frames"; tmp.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(D / "video" / f"{a.id}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    t_ax = (np.arange(f0, f1) - f0) / fps
    a0, a1 = (s * fps - f0) / fps, ((e + 1) * fps - f0) / fps
    fig = plt.figure(figsize=(16, 5.4), dpi=80)
    for i in range(f1 - f0):
        ok, fr = cap.read()
        fig.clf()
        ax0 = fig.add_axes([0.0, 0.05, 0.34, 0.9]); ax0.axis("off")
        if ok:
            b = bj["tight_clean"][f0 + i]
            if bj["present"][f0 + i] and b is not None:
                x1, y1, x2, y2 = map(int, b); cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            ax0.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        ax0.set_title(f"video {a.id}  {ann['element']}  frame {f0 + i}  t={(f0 + i) / fps:.2f}s", fontsize=10)
        ax = fig.add_axes([0.34, 0.02, 0.33, 0.96], projection="3d")
        j = J[i]
        for c, p in enumerate(PARENTS):
            if p >= 0:
                ax.plot([j[c, 0], j[p, 0]], [j[c, 2], j[p, 2]], [j[c, 1], j[p, 1]], color="tab:red" if c in LEFT else "tab:blue", lw=2.5)
        ax.plot(J[:i + 1, 0, 0], J[:i + 1, 0, 2], J[:i + 1, 0, 1] * 0 + ground, color="gray", lw=1)   # pelvis path on the ground
        gx = np.linspace(ctr[0] - half, ctr[0] + half, 9); gz = np.linspace(ctr[1] - half, ctr[1] + half, 9)
        for x in gx: ax.plot([x, x], [gz[0], gz[-1]], [ground, ground], color="0.85", lw=0.6)
        for z in gz: ax.plot([gx[0], gx[-1]], [z, z], [ground, ground], color="0.85", lw=0.6)
        ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half); ax.set_zlim(ground - 0.05, ground + 2.3)
        ax.set_box_aspect((2 * half, 2 * half, 2.35)); ax.view_init(elev=12, azim=-60)       # equal metres on all axes
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("up (m)")
        ax.set_title("GVHMR world (one segment, Y up)", fontsize=10)
        ax2 = fig.add_axes([0.71, 0.12, 0.27, 0.78])
        ax2.axvspan(a0, a1, color="0.92", label="annotated element time")
        ax2.plot(t_ax, (pel - ground) * 100, label="pelvis height", color="tab:blue")
        ax2.plot(t_ax, (foot - ground) * 100, label="lowest foot height", color="tab:orange")
        ax2.axvline(t_ax[i], color="k", lw=1)
        ax2.set_xlabel("seconds"); ax2.set_ylabel("cm above ground (5th pct. of lowest foot)")
        ax2.legend(fontsize=8, loc="upper right"); ax2.grid(alpha=0.3)
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
