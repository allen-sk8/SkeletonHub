"""Model registry: label, conda env python, extra environment, GPU rule and how to construct the detector."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path

ENVS = Path("/home/allen/miniconda3/envs")


@dataclass
class ModelSpec:
    key: str
    label: str          # label used in the video grid
    env: str            # conda env name
    mesh: str           # smpl / mhr / skeleton
    ctor: str           # "module:Class"
    extra_env: dict = field(default_factory=dict)
    license_note: str = ""

    @property
    def python(self) -> str:
        return str(ENVS / self.env / "bin" / "python")


SPECS = {s.key: s for s in [
    ModelSpec("hybrik", "HybrIK (SMPL)", "skeleton_env", "smpl", "hmr_zoo.adapters:HybrIKDetector",
              license_note="HybrIK code Apache-2.0-like; SMPL non-commercial"),
    ModelSpec("gvhmr", "GVHMR (SMPL)", "skeleton_env", "smpl", "hmr_zoo.adapters:GVHMRDetector",
              license_note="GVHMR research license; SMPL non-commercial"),
    ModelSpec("wham", "WHAM (SMPL)", "wham", "smpl", "detectors.wham_detector:WHAMDetector",
              license_note="WHAM research license; SMPL non-commercial"),
    ModelSpec("instanthmr", "InstantHMR (MHR)", "skeleton_env", "mhr", "hmr_zoo.adapters:InstantHMRDetector",
              license_note="check upstream license; MHR (Meta) terms"),
    ModelSpec("nlf", "NLF (SMPL)", "finefs_pose", "smpl", "detectors.nlf_detector:NLFDetector",
              license_note="research only; SMPL non-commercial"),
    ModelSpec("sam3db", "SAM 3D Body (MHR)", "finefs_pose", "mhr", "detectors.sam3db_detector:SAM3DBodyDetector",
              license_note="SAM License (Meta), research"),
    ModelSpec("onlinehmr", "OnlineHMR (SMPL)", "onlinehmr", "smpl", "detectors.onlinehmr_detector:OnlineHMRDetector",
              extra_env={"LD_LIBRARY_PATH": str(ENVS / "onlinehmr" / "lib")},
              license_note="official repo has NO LICENSE file: research use only"),
    ModelSpec("rtmpose3d", "RTMW3D-x (skeleton)", "skeleton_env", "skeleton", "hmr_zoo.adapters:RTMPose3DDetector",
              license_note="Apache-2.0 (mmpose)"),
]}

# the default list used by compare.py (finefs Stage 2 order + OnlineHMR; rtmpose3d is skeleton-only)
DEFAULT_MODELS = ["hybrik", "gvhmr", "wham", "instanthmr", "nlf", "sam3db", "onlinehmr"]


def build(key: str, device: str = "cuda:0"):
    mod, cls = SPECS[key].ctor.split(":")
    return getattr(importlib.import_module(mod), cls)(device=device)
