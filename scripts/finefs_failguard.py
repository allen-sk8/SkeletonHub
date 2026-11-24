"""Per-video failure handling for the FineFS v2 full batch (shared by the tracking, 3D and export scripts).

A video that raises is recorded in <fail_dir>/<id>.<stage>.txt (traceback) and skipped from then on; the script goes on
with the next video. Exit codes the batch supervisor (scripts/run_finefs_v2_full.sh) acts on:
  3  CUDA out of memory: the supervisor checks whether another user's process is on the card and moves to another card
  4  MAX_CONSEC videos in a row failed: something is broken, the whole batch stops
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

MAX_CONSEC = 3


class FailGuard:
    def __init__(self, fail_dir, stage: str):
        self.dir = Path(fail_dir) if fail_dir else None
        self.stage = stage
        self.consec = 0

    def marker(self, vid):
        return self.dir / f"{vid}.{self.stage}.txt"

    def skip(self, vid) -> bool:
        if self.dir and self.marker(vid).exists():
            print(f"[{self.stage}] {vid}: failed before ({self.marker(vid)}), skipped", flush=True)
            return True
        return False

    def ok(self):
        self.consec = 0

    def failed(self, vid, exc: BaseException):
        oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        tb = traceback.format_exc()
        print(f"[{self.stage}] {vid}: FAILED {'(CUDA out of memory) ' if oom else ''}{type(exc).__name__}: {exc}", flush=True)
        if self.dir is None:
            raise exc
        self.dir.mkdir(parents=True, exist_ok=True)
        self.marker(vid).write_text(f"{time.strftime('%F %T')} {'oom' if oom else 'error'}\n{tb}")
        if oom:
            sys.exit(3)
        self.consec += 1
        if self.consec >= MAX_CONSEC:
            print(f"[{self.stage}] {self.consec} videos in a row failed, exiting with code 4", flush=True)
            sys.exit(4)
