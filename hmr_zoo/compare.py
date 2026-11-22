"""Run all models on one or more clip directories, then render compare_2d.mp4 + metrics.

    python -m hmr_zoo.compare --clips DIR1,DIR2 --gpu 0 [--models hybrik,gvhmr,...] [--skip-existing] [--render-only]

Each model runs in its own conda env as a subprocess (hmr_zoo.run_model), once for all clips (model loaded once).
Models that fail are reported and skipped; the others still go into the video. The GPU index is the nvidia-smi
index (CUDA_DEVICE_ORDER=PCI_BUS_ID is set here). Before every model the card is checked: if anything other than
this run uses it (>200 MiB used by a foreign process) the run stops (shared lab machine).

Run with any python that has numpy + opencv + pillow (the render step), e.g. the `reidtrack` or `skeleton_env` env.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hmr_zoo import models as M  # noqa: E402


def gpu_used_mb(idx: str) -> float:
    out = subprocess.check_output(["nvidia-smi", "-i", idx, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return float(out.strip().splitlines()[0])


def foreign_procs(idx: str) -> list[str]:
    """PIDs on this GPU not descended from this process (nvidia-smi --query-compute-apps by gpu uuid)."""
    uuid = subprocess.check_output(["nvidia-smi", "-i", idx, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    pids = [l.split(",")[1].strip() for l in out.splitlines() if l.split(",")[0].strip() == uuid]
    mine = set()
    cur = os.getpid()
    for p in pids:
        pp = p
        while pp not in ("", "0", "1"):
            if pp == str(cur):
                mine.add(p)
                break
            try:
                pp = subprocess.check_output(["ps", "-o", "ppid=", "-p", pp], text=True).strip()
            except subprocess.CalledProcessError:
                break
    return [p for p in pids if p not in mine]


def run_model(key: str, clips: list[Path], gpu: str, log: Path, warmup: bool) -> bool:
    spec = M.SPECS[key]
    env = os.environ.copy()
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(ROOT),
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    env.pop("VIRTUAL_ENV", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(spec.extra_env)
    cmd = ["nice", "-n", "10", spec.python, "-u", "-m", "hmr_zoo.run_model", "--model", key,
           "--clips", ",".join(str(c) for c in clips)]
    if not warmup:
        cmd.append("--no-warmup")
    t0 = time.time()
    print(f"[compare] {key}: {len(clips)} clips on GPU {gpu} ({spec.env})", flush=True)
    with log.open("a") as f:
        f.write(f"\n===== {key} {time.strftime('%F %T')} =====\n")
        f.flush()
        for attempt in range(3):
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
            if rc != 126:  # 126 = "permission denied" on the env's python: seen as a transient NFS glitch, retry
                break
            f.write(f"rc=126 (python not executable), retry {attempt + 1}\n")
            time.sleep(10)
    print(f"[compare] {key}: rc={rc} in {time.time() - t0:.0f}s", flush=True)
    return rc == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="comma-separated clip directories")
    ap.add_argument("--models", default=",".join(M.DEFAULT_MODELS))
    ap.add_argument("--gpu", default="", help="nvidia-smi GPU index (required unless --render-only)")
    ap.add_argument("--skip-existing", action="store_true", help="skip model/clip pairs that already have an npz")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--log", default="")
    a = ap.parse_args()

    clips = [Path(c) for c in a.clips.split(",") if c]
    models = [m for m in a.models.split(",") if m]
    log = Path(a.log) if a.log else ROOT / "outputs" / "hmr_zoo" / "compare.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status = {}
    if not a.render_only:
        if not a.gpu:
            raise SystemExit("--gpu required")
        for m in models:
            todo = [c for c in clips if not (a.skip_existing and (c / f"{m}.npz").exists())]
            if not todo:
                status[m] = "skipped (exists)"
                continue
            fp = foreign_procs(a.gpu)
            if fp:  # a just-exited process of ours can linger in nvidia-smi for a moment: look again
                time.sleep(5)
                fp = [p for p in foreign_procs(a.gpu) if Path(f"/proc/{p}").exists()]
            if fp:
                raise SystemExit(f"GPU {a.gpu} has foreign processes {fp}; stopping (shared machine)")
            ok = run_model(m, todo, a.gpu, log, warmup=not a.no_warmup)
            status[m] = "ok" if ok else "FAILED (see log)"
        print(json.dumps(status, indent=1))

    from hmr_zoo.render import render_clip

    all_rows = []
    for c in clips:
        have = [m for m in models if (c / f"{m}.npz").exists()]
        if not have:
            print(f"[compare] {c.name}: no predictions, skip render")
            continue
        all_rows += render_clip(c, have)
    if all_rows:
        out = clips[0].parent / "metrics_all.csv" if len({c.parent for c in clips}) == 1 else ROOT / "outputs/hmr_zoo/metrics_all.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"[compare] metrics -> {out}")


if __name__ == "__main__":
    main()
