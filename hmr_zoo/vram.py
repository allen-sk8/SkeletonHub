"""Process-level GPU memory peak monitor (nvidia-smi pid query; fallback: visible-GPU delta).

Same logic as finefs_preproc/scripts/stage2_infer_skeletonhub.py (needed because onnxruntime and
subprocess-style allocations are invisible to torch.cuda.max_memory_allocated).
"""

from __future__ import annotations

import os
import subprocess
import threading


class ProcessVramMonitor:
    def __init__(self, baseline_mb: float | None = None, interval: float = 0.12) -> None:
        self.pid = str(os.getpid())
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        self.gpu_idx = vis if vis.isdigit() else "0"
        self.interval = interval
        self.baseline_mb = self._query_gpu_total_mb() if baseline_mb is None else float(baseline_mb)
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _query_gpu_total_mb(self) -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-i", self.gpu_idx, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2.0,
            ).strip()
            return float(out.splitlines()[0].strip()) if out else 0.0
        except Exception:
            return 0.0

    def _sample_once(self) -> float:
        pid_mb = 0.0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=2.0,
            )
            for line in out.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0] == self.pid:
                    pid_mb += float(parts[1])
        except Exception:
            pass
        if pid_mb > 0.0:
            return pid_mb
        return max(0.0, self._query_gpu_total_mb() - self.baseline_mb)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._sample_once())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_mb = self._sample_once()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_mb = max(self.peak_mb, self._sample_once())

    @property
    def peak_gb(self) -> float:
        return self.peak_mb / 1024.0
