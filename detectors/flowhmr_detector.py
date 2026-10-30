"""
FlowHMR 3D Skeleton Detector & Estimator (SMPL24, OpenPose BODY25, H36M17, SMPL-H 52j)

Example Usage:
    python detectors/flowhmr_detector.py --input data/video/8_front_9_cam2.mp4 --device cuda:2 --vis
    # Skip inference and only convert an existing FlowHMR result:
    python detectors/flowhmr_detector.py --input data/video/x.mp4 --from-npz path/to/flowhmr_latest.npz

⚠️ Known Limitation (tested 2026-10-30 on FineFS 5s clips, 13 videos):
    - Ordinary motion (e.g. data/video/8_front_9_cam2.mp4): good.
    - Low-rotation skating (1A, 2A+2Lo, 1A+2T): not evaluated in detail.
    - High-rotation figure skating jumps (3A, 4T, 4S, 4Lz, 4F, 4Lo, 4Lz+3T, 4S+3T): FAILS.
      The mid-air rotation collapses into a "frozen" pose (the body stops rotating while airborne).
      Do not use FlowHMR for fast-spinning skating jumps; prefer GVHMR / HybrIK there.

Technical Details & Reference Sources:
    - Reference Sources:
      * FlowHMR inference entry: `ReasonMotion_Predictor/baselines/flowhmr/run_v2m_demo_generation.py`
      * FlowHMR pipeline:        `.../flowhmr/flowhmr/pipeline/v2m_inference.py` (YOLOX ➔ VGGT-Omega camera ➔ SAM 3D Body feature ➔ FlowHMR)
      * Joint regressors:        `common_models/regressor/J_regressor_body25.npy`, `common_models/body_models/smpl/J_regressor_h36m.npy`
      * SMPL-H FK & 24j mapping: `utils/smpl/handler.py`, `converters/smplh_to_smpl_24j.py`
      * Centering / grounding:   `detectors/gvhmr_detector.py`

    - Environment isolation:
      FlowHMR needs its own conda env (`flowhmr`: Python 3.10, PyTorch 2.5.1+cu118, gated VGGT-Omega / SAM 3D Body weights),
      which cannot share imports with `skeleton_env`. This wrapper therefore runs FlowHMR as a SUBPROCESS using the
      flowhmr env's python (absolute path, `VIRTUAL_ENV` unset, `PYTHONNOUSERSITE=1`) and then converts the saved `.npz`
      inside the current (skeleton_env) process. Override locations with env vars:
        FLOWHMR_ROOT   (default: /home/allen/ReasonMotion_Predictor/baselines/flowhmr)
        FLOWHMR_PYTHON (default: /home/allen/miniconda3/envs/flowhmr/bin/python)
      The system ffmpeg (4.4.2) lacks `-fps_mode`, so FFMPEG is pointed at the `imageio_ffmpeg` 7.x binary inside the flowhmr env.

    - Data Flow:
      1. Video ➔ (subprocess) FlowHMR ➔ `<FLOWHMR_ROOT>/output/<stem>/flowhmr_latest.npz` (cached, reused unless --overwrite)
         npz keys: poses (T,52,3) SMPL-H axis-angle, trans (T,3), betas (1,16), local_joints_positions (T,52,3) ...
         Output is natively Y-up, meters, 30 fps (video is transcoded to 30 fps first).
      2. Save SMPL-H parameters as `data/smpl/smplh/<stem>_flowhmr_smplh.pkl` (poses 156D, betas, trans, gender='neutral'),
         so every existing `converters/smplh_to_*.py` works on it.
      3. SMPLHandler FK (neutral SMPL-H) ➔ 6890 vertices / 52 joints.
      4. For SMPL24:   52j FK joints, indices [:22] + [22] (L hand) + [37] (R hand)  ➔ `_flowhmr_24j.npy`
         For BODY25:   vertices x J_regressor_body25  ➔ `_flowhmr_body25.npy`
         For H36M17:   vertices x J_regressor_h36m    ➔ `_flowhmr_17j.npy`
         For SMPL-H 52j: FK joints ➔ `_flowhmr_52j.npy`
      5. Centering: shift X and Z so the first frame's pelvis (MidHip for BODY25) is at the origin.
         Grounding: shift Y so the lowest joint of frame 0 is at Y = 0 (disable with --disable-rebase).

    - Not provided (unlike GVHMR/HybrIK): 2D skeleton / SMPL mesh overlays on the video. FlowHMR's result has no in-camera
      parameters (the VGGT camera is cached separately in `output/<stem>/<stem>_vggt_camera.npz`).
"""
import os
import sys
import shutil
import pickle
import argparse
import subprocess
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

FLOWHMR_ROOT = os.environ.get("FLOWHMR_ROOT", "/home/allen/ReasonMotion_Predictor/baselines/flowhmr")
FLOWHMR_PYTHON = os.environ.get("FLOWHMR_PYTHON", "/home/allen/miniconda3/envs/flowhmr/bin/python")

LIMITATION_NOTE = ("⚠️ FlowHMR is unreliable on high-rotation figure skating jumps "
                   "(mid-air rotation collapses into a frozen pose). Fine for ordinary motion.")


def run_flowhmr(video_path, device="cuda:0", vggt_interval=1, overwrite=False):
    """
    Run FlowHMR on one video in its own conda env (subprocess) and return the path of the result .npz.
    Results are cached under <FLOWHMR_ROOT>/output/<video_stem>/flowhmr_latest.npz.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    result_npz = os.path.join(FLOWHMR_ROOT, "output", stem, "flowhmr_latest.npz")
    if os.path.exists(result_npz) and not overwrite:
        print(f"♻️ Reusing cached FlowHMR result: {result_npz}")
        return result_npz

    if not os.path.exists(FLOWHMR_PYTHON):
        raise FileNotFoundError(f"❌ FlowHMR python not found: {FLOWHMR_PYTHON} (set FLOWHMR_PYTHON)")
    env_dir = os.path.dirname(os.path.dirname(FLOWHMR_PYTHON))
    ffmpeg_dir = os.path.join(env_dir, "lib", "python3.10", "site-packages", "imageio_ffmpeg", "binaries")
    ffmpeg_bin = None
    if os.path.isdir(ffmpeg_dir):
        cands = [f for f in os.listdir(ffmpeg_dir) if f.startswith("ffmpeg")]
        if cands:
            ffmpeg_bin = os.path.join(ffmpeg_dir, sorted(cands)[-1])

    # Clean environment: another venv on PATH / ~/.local site-packages would shadow the flowhmr env
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = os.path.dirname(FLOWHMR_PYTHON) + os.pathsep + env.get("PATH", "")
    if ffmpeg_bin:
        env["FFMPEG"] = ffmpeg_bin

    cmd = [FLOWHMR_PYTHON, "-u", "run_v2m_demo_generation.py", os.path.abspath(video_path),
           "--device", device, "--vggt_interval", str(vggt_interval)]
    if overwrite:
        cmd.append("--overwrite")
    print(f"🚀 Running FlowHMR (device={device}): {' '.join(cmd)}")
    subprocess.run(cmd, cwd=FLOWHMR_ROOT, env=env, check=True)

    if not os.path.exists(result_npz):
        raise RuntimeError(f"❌ FlowHMR finished but result not found: {result_npz}")
    return result_npz


def load_flowhmr_npz(npz_path):
    """Load a FlowHMR result and convert it to SkeletonHub SMPL-H (156D) parameter dict."""
    d = np.load(npz_path, allow_pickle=True)
    poses = d["poses"].astype(np.float32)          # (T, 52, 3) axis-angle
    T = poses.shape[0]
    return {
        "poses": poses.reshape(T, -1),             # (T, 156)
        "betas": d["betas"].astype(np.float32).reshape(1, -1),
        "trans": d["trans"].astype(np.float32),    # (T, 3)
        "gender": "neutral",
        "mocap_framerate": int(d["mocap_framerate"]) if "mocap_framerate" in d.files else 30,
    }


def ensure_smplh_neutral_npz():
    """
    smplx looks for `smplh/SMPLH_NEUTRAL.npz`, but the neutral model (16 shape coefficients, as used by FlowHMR)
    is stored as `smplh/neutral/model.npz` and lacks the hand-PCA fields smplx insists on
    (`hands_components{l,r}`, `hands_mean{l,r}`; unused here because use_pca=False).
    Without this file SMPLHandler silently falls back to the female model and truncates betas to 10.
    The derived file is generated once next to the original (common_models/ is git-ignored).
    """
    smplh_dir = os.path.join(project_root, "common_models", "body_models", "smplh")
    src = os.path.join(smplh_dir, "neutral", "model.npz")
    dst = os.path.join(smplh_dir, "SMPLH_NEUTRAL.npz")
    if os.path.exists(src) and not os.path.exists(dst):
        data = dict(np.load(src, allow_pickle=True))
        for side in ("l", "r"):
            data.setdefault(f"hands_components{side}", np.eye(45, dtype=np.float32))
            data.setdefault(f"hands_mean{side}", np.zeros(45, dtype=np.float32))
        np.savez(dst, **data)
        print(f"🔧 Generated {dst} from neutral/model.npz")


def smplh_params_to_skeletons(params, run_smpl24=True, run_body25=True, run_h36m17=True, run_52j=True, rebase=True):
    """
    SMPL-H parameters ➔ requested skeleton formats (Y-up, meters, centered at frame-0 pelvis XZ, optionally grounded).
    """
    from utils.smpl.handler import SMPLHandler

    ensure_smplh_neutral_npz()
    # FlowHMR predicts 16 shape coefficients for SMPL-H neutral (model.npz)
    handler = SMPLHandler(model_type="smplh", num_betas=params["betas"].shape[1])
    poses, betas, trans, gender = params["poses"], params["betas"], params["trans"], params["gender"]
    out = {"smpl24": None, "body25": None, "h36m17": None, "smplh52": None}

    def finalize(joints, root_idx):
        joints = joints.copy()
        joints[..., 0] -= joints[0, root_idx, 0]
        joints[..., 2] -= joints[0, root_idx, 2]
        if rebase:
            joints[..., 1] -= joints[0, :, 1].min()
        return joints.astype(np.float32)

    # Grounding offset is computed per format from its own frame 0, consistent with gvhmr_detector.py
    if run_smpl24 or run_52j:
        # smplx returns 73 points (52 joints + face/fingertip landmarks); keep the 52 SMPL-H joints
        joints_52 = handler.params_to_joints(poses, betas, trans, gender)[:, :52]  # (T, 52, 3)
        if run_52j:
            out["smplh52"] = finalize(joints_52, 0)
        if run_smpl24:
            # [:22] body + [22] left hand + [37] right hand (see converters/smplh_to_smpl_24j.py)
            joints_24 = np.concatenate([joints_52[:, :22], joints_52[:, 22:23], joints_52[:, 37:38]], axis=1)
            out["smpl24"] = finalize(joints_24, 0)

    if run_body25 or run_h36m17:
        vertices, _ = handler.params_to_vertices(poses, betas, trans, gender)  # (T, 6890, 3)
        if run_body25:
            reg25 = np.load(os.path.join(project_root, "common_models", "regressor", "J_regressor_body25.npy"))
            out["body25"] = finalize(np.matmul(reg25[None], vertices), 8)  # MidHip = 8
        if run_h36m17:
            reg17 = np.load(os.path.join(project_root, "common_models", "body_models", "smpl", "J_regressor_h36m.npy"))
            out["h36m17"] = finalize(np.matmul(reg17[None], vertices), 0)
    return out


def main():
    parser = argparse.ArgumentParser(description="FlowHMR SMPL24, BODY25, H36M17 and SMPL-H 52j Detector Wrapper")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output-smpl24", help="Path to output SMPL24 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-body25", help="Path to output BODY25 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-h36m17", help="Path to output H36M17 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-smplh52", help="Path to output SMPL-H 52j .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-smplh-pkl", help="Path to output SMPL-H (156D) .pkl. If omitted, uses auto-suffixing.")
    parser.add_argument("--device", default="cuda:0", help="GPU device (default: cuda:0). Default settings peak at ~30 GB; "
                        "on 24 GB cards add --vggt-interval 10")
    parser.add_argument("--vggt-interval", type=int, default=1, help="Run camera estimation on every N-th frame (lower GPU memory)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute FlowHMR even if a cached result exists")
    parser.add_argument("--from-npz", help="Skip inference and convert this existing FlowHMR result .npz")
    parser.add_argument("--disable-rebase", action="store_true", help="Disable rebasing lowest keypoint height to 0")
    parser.add_argument("--format", choices=['smpl24', 'body25', 'h36m17', 'smplh52', 'all'], default='all',
                        help="Output skeleton format (default: all)")
    parser.add_argument("--vis", action="store_true", help="Automatically run 3D skeleton joint visualizer scripts")
    args = parser.parse_args()

    if not os.path.exists(args.input) and not args.from_npz:
        print(f"❌ Input video file not found: {args.input}")
        sys.exit(1)

    run_smpl24 = args.format in ['smpl24', 'all']
    run_body25 = args.format in ['body25', 'all']
    run_h36m17 = args.format in ['h36m17', 'all']
    run_52j = args.format in ['smplh52', 'all']

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    def default_path(sub, suffix, given):
        if given:
            return given
        d = os.path.join(project_root, "data", *sub)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{base_name}_{suffix}")

    out_paths = {
        "smpl24": default_path(["smpl_joints", "samples_24j"], "flowhmr_24j.npy", args.output_smpl24),
        "body25": default_path(["body25"], "flowhmr_body25.npy", args.output_body25),
        "h36m17": default_path(["h36m17"], "flowhmr_17j.npy", args.output_h36m17),
        "smplh52": default_path(["smpl_joints", "samples_52j"], "flowhmr_52j.npy", args.output_smplh52),
    }
    pkl_path = default_path(["smpl", "smplh"], "flowhmr_smplh.pkl", args.output_smplh_pkl)

    try:
        npz_path = args.from_npz or run_flowhmr(args.input, args.device, args.vggt_interval, args.overwrite)
        params = load_flowhmr_npz(npz_path)

        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(params, f)
        print(f"✅ SMPL-H parameters saved to: {pkl_path} (poses {params['poses'].shape}, trans {params['trans'].shape})")

        results = smplh_params_to_skeletons(params, run_smpl24, run_body25, run_h36m17, run_52j,
                                            rebase=not args.disable_rebase)
        for key, label in [("smpl24", "SMPL24"), ("body25", "BODY25"), ("h36m17", "H36M17"), ("smplh52", "SMPL-H 52j")]:
            if results[key] is not None:
                os.makedirs(os.path.dirname(out_paths[key]), exist_ok=True)
                np.save(out_paths[key], results[key])
                print(f"✅ {label} output saved to: {out_paths[key]} (Shape: {results[key].shape})")

        if args.vis:
            vis_map = {"smpl24": "vis_smpl_joints.py", "body25": "vis_body25_joints.py",
                       "h36m17": "vis_smpl_joints.py", "smplh52": "vis_smpl_joints.py"}
            for key, script in vis_map.items():
                if results[key] is None:
                    continue
                vis_script = os.path.join(project_root, "visualizers", script)
                print(f"🎬 Visualizing {key} using {script}...")
                subprocess.run([sys.executable, vis_script, out_paths[key], "--fps", "30"])

        print(LIMITATION_NOTE)

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
