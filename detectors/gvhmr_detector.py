"""
GVHMR 3D Skeleton Detector & Estimator (SMPL24, OpenPose BODY25, and H36M17)

Example Usage:
    python detectors/gvhmr_detector.py --input data/video.mp4 --vis

Technical Details & Reference Sources:
    - Reference Sources:
      * GVHMR Demo Execution: `external/gvhmr/tools/demo/demo.py`
      * GVHMR PL Demo Model: `external/gvhmr/hmr4d/model/gvhmr/gvhmr_pl_demo.py`
      * Joint Regressors & Body Models: `external/gvhmr/hmr4d/utils/smplx_utils.py`
      * Coordinate & Scale Alignment: `converters/HybrIK_to_joints_24j.py` and `detectors/hybrik_detector.py`
    
    - Data Flow:
      1. Video Input ➔ Runs YOLOv8 tracking, ViTPose, and ViT feature extractors.
      2. Runs GVHMR model to predict SMPL-X global and camera parameters.
      3. For SMPL24:
         - Regress standard SMPL vertices (6890) to 24 joints using `smpl_neutral_J_regressor.pt`.
         - GVHMR outputs are natively Y-up (meters).
         - Centering: Center X and Z around the first frame's pelvis (joint 0).
         - Grounding: Offset Y so that the lowest joint of frame 0 is at Y = 0.
         - Save as `_gvhmr_24j.npy`.
      4. For OpenPose BODY25:
         - Regress standard SMPL vertices (6890) to 25 joints using `J_regressor_body25.npy`.
         - Centering: Center X and Z around the first frame's MidHip (joint 8).
         - Grounding: Offset Y so that the lowest joint of frame 0 is at Y = 0.
         - Save as `_gvhmr_body25.npy`.
      5. For H36M17:
         - Regress standard SMPL vertices (6890) to 17 joints using `J_regressor_h36m.npy`.
         - Centering: Center X and Z around the first frame's pelvis (joint 0).
         - Grounding: Offset Y so that the lowest joint of frame 0 is at Y = 0.
         - Save as `_gvhmr_17j.npy`.
"""
import os
import sys
import argparse
import torch
import numpy as np
import tempfile
import cv2
import subprocess
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import GVHMR components
from hmr4d import PROJ_ROOT
import hmr4d.model.gvhmr.gvhmr_pl_demo
from hydra import initialize_config_module, compose
from hydra.core.global_hydra import GlobalHydra
from hmr4d.configs import register_store_gvhmr
from hydra.utils import instantiate
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, perspective_projection
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.video_io_utils import get_video_lwh, get_writer, get_video_reader
from hmr4d.utils.preproc import Tracker, VitPoseExtractor, Extractor, SimpleVO
from hmr4d.utils.vis.renderer import Renderer

# --- OpenPose BODY25 skeleton topology ---
BODY_25_CHAIN = [
    [8, 1, 0, 15, 17],        # Spine -> Neck -> Nose -> REye -> REear
    [0, 16, 18],              # Nose -> LEye -> LEar
    [1, 2, 3, 4],             # Neck -> RShoulder -> RElbow -> RWrist
    [1, 5, 6, 7],             # Neck -> LShoulder -> LElbow -> LWrist
    [8, 9, 10, 11, 22, 23],   # MidHip -> RHip -> RKnee -> RAnkle -> RBigToe -> RSmallToe
    [11, 24],                 # RAnkle -> RHeel
    [8, 12, 13, 14, 19, 20],  # MidHip -> LHip -> LKnee -> LAnkle -> LBigToe -> LSmallToe
    [14, 21]                  # LAnkle -> LHeel
]

BODY_25_COLORS = [
    (0, 255, 0),   # Spine -> Neck -> Nose (Green)
    (0, 255, 0),   # Nose -> LEye -> LEar (Green)
    (0, 0, 255),   # Right arm (Red)
    (255, 0, 0),   # Left arm (Blue)
    (0, 0, 255),   # Right leg (Red)
    (0, 0, 255),   # Right heel (Red)
    (255, 0, 0),   # Left leg (Blue)
    (255, 0, 0)    # Left heel (Blue)
]

# --- Standard SMPL 24 joints topology ---
SMPL_24_CHAIN = [
    [0, 2, 5, 8, 11],         # Right leg
    [0, 1, 4, 7, 10],         # Left leg
    [0, 3, 6, 9, 12, 15],     # Spine & Head
    [9, 14, 17, 19, 21, 23],  # Right arm
    [9, 13, 16, 18, 20, 22]   # Left arm
]

SMPL_24_COLORS = [
    (0, 0, 255),   # Right leg (Red)
    (255, 0, 0),   # Left leg (Blue)
    (0, 255, 0),   # Spine & Head (Green)
    (0, 0, 255),   # Right arm (Red)
    (255, 0, 0)    # Left arm (Blue)
]

# --- Human3.6M 17 joints topology ---
H36M_17_CHAIN = [
    [0, 1, 2, 3],         # Left leg
    [0, 4, 5, 6],         # Right leg
    [0, 7, 8, 9, 10],     # Spine & Head
    [8, 11, 12, 13],      # Left arm
    [8, 14, 15, 16]       # Right arm
]

H36M_17_COLORS = [
    (255, 0, 0),   # Left leg (Blue)
    (0, 0, 255),   # Right leg (Red)
    (0, 255, 0),   # Spine & Head (Green)
    (255, 0, 0),   # Left arm (Blue)
    (0, 0, 255)    # Right arm (Red)
]


def vis_skeleton_video(video_path, joints_2d, format_name, output_video_path, fps=30):
    """
    Visualize 2D projected joints/bones overlaid back onto the input video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = orig_fps if orig_fps > 0 else 30
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if format_name == 'body25':
        chain = BODY_25_CHAIN
        colors = BODY_25_COLORS
    elif format_name == 'h36m17':
        chain = H36M_17_CHAIN
        colors = H36M_17_COLORS
    else:  # 'smpl24'
        chain = SMPL_24_CHAIN
        colors = SMPL_24_COLORS
        
    t = 0
    total_frames = int(joints_2d.shape[0])
    
    print(f"🎬 Rendering skeleton overlay video to: {output_video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret or t >= total_frames:
            break
            
        frame_joints = joints_2d[t]
        
        # Draw bones
        for limb, color in zip(chain, colors):
            for i in range(len(limb) - 1):
                p1 = tuple(frame_joints[limb[i]].astype(int))
                p2 = tuple(frame_joints[limb[i+1]].astype(int))
                if (0 <= p1[0] < width and 0 <= p1[1] < height and
                    0 <= p2[0] < width and 0 <= p2[1] < height):
                    cv2.line(frame, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                    
        # Draw joints
        for pt in frame_joints:
            p = tuple(pt.astype(int))
            if 0 <= p[0] < width and 0 <= p[1] < height:
                cv2.circle(frame, p, radius=4, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(frame, p, radius=5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                
        out.write(frame)
        t += 1
        
    cap.release()
    out.release()
    
    # H.264 optimization using ffmpeg
    print(f"🔄 Optimizing video encoding (H.264)...")
    temp_path = output_video_path.replace(".mp4", "_temp.mp4")
    if os.path.exists(output_video_path):
        try:
            os.rename(output_video_path, temp_path)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", temp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", output_video_path]
            subprocess.run(cmd, check=True)
            os.remove(temp_path)
        except Exception as e:
            print(f"⚠️ Video optimization failed: {e}")
            if os.path.exists(temp_path):
                os.rename(temp_path, output_video_path)


class GVHMRDetector:
    def __init__(self, device='cuda:0'):
        """
        Initialize the GVHMR detector and load models.
        """
        self.device = device
        
        print("⚙️ Initializing GVHMR Detector...")
        
        # Configure and compose via Hydra Zen
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
            
        register_store_gvhmr()
        initialize_config_module(version_base="1.3", config_module="hmr4d.configs")
        cfg = compose(config_name="demo", overrides=["video_name=dummy_name", "static_cam=True", "use_dpvo=False"])
        
        # Instantiate model
        self.model = instantiate(cfg.model, _recursive_=False)
        ckpt_path = PROJ_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
        self.model.load_pretrained_model(ckpt_path)
        self.model = self.model.eval().to(self.device)
        
        # Setup SMPL-X to SMPL mapping & Joint Regressors
        self.smplx = make_smplx("supermotion").to(self.device)
        self.smplx2smpl = torch.load(PROJ_ROOT / "hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(self.device)
        self.J_regressor_24 = torch.load(PROJ_ROOT / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(self.device)
        
        # Load BODY25 joint regressor
        regressor_25_path = os.path.join(project_root, 'external', 'EasyMocap', 'data', 'smplx', 'J_regressor_body25.npy')
        self.J_regressor_25 = torch.from_numpy(np.load(regressor_25_path)).float().to(self.device)
        
        # Load H36M17 joint regressor
        regressor_17_path = os.path.join(project_root, 'external', 'WHAM', 'dataset/body_models/J_regressor_h36m.npy')
        self.J_regressor_17 = torch.from_numpy(np.load(regressor_17_path)).float().to(self.device)
        
    def detect_video(self, video_path, static_cam=True, rebase=True, run_smpl24=True, run_body25=True, run_h36m17=True):
        """
        Run GVHMR pipeline on a video and return estimated coordinates.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Input video file not found: {video_path}")
            
        length, width, height = get_video_lwh(video_path)
        
        # 1. Run Preprocessing Extractors
        print("📐 Running Preprocessing Extractors...")
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        del tracker
        
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        del vitpose_extractor
        
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        del extractor
        
        # Handle camera motion
        if not static_cam:
            print("📷 Computing camera trajectories...")
            simple_vo = SimpleVO(video_path, scale=0.5, step=8, method="sift", f_mm=None)
            vo_results = simple_vo.compute()
            R_w2c = torch.from_numpy(vo_results[:, :3, :3]).float()
        else:
            R_w2c = torch.eye(3).repeat(length, 1, 1).float()
            
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1).float()
        
        # 2. Package data for model inference
        data = {
            "length": torch.tensor(length),
            "bbx_xys": bbx_xys,
            "kp2d": vitpose,
            "K_fullimg": K_fullimg,
            "cam_angvel": compute_cam_angvel(R_w2c),
            "f_imgseq": vit_features,
        }
        
        # 3. Model Inference
        print("🧠 Running GVHMR Model Inference...")
        pred = self.model.predict(data, static_cam=static_cam)
        
        # 4. Extract vertices and joints
        print("📐 Reconstructing 3D Skeletons...")
        output_dict = {
            'smpl24': None,
            'body25': None,
            'h36m17': None,
            'joints_24_cam': None,
            'joints_25_cam': None,
            'joints_17_cam': None,
            'vertices_glob': None,
            'vertices_cam': None,
            'faces': make_smplx("smpl").faces,
            'K_fullimg': K_fullimg.numpy()
        }
        
        def to_device(batch, device):
            if isinstance(batch, dict):
                return {k: to_device(v, device) for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                return batch.to(device)
            return batch
            
        # Reconstruct global space (Y-up world)
        smpl_params_global = to_device(pred["smpl_params_global"], self.device)
        smplx_out_global = self.smplx(**smpl_params_global)
        verts_glob = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smplx_out_global.vertices])
        output_dict['vertices_glob'] = verts_glob.cpu().numpy()
        
        # Reconstruct camera space (Y-down camera)
        smpl_params_incam = to_device(pred["smpl_params_incam"], self.device)
        smplx_out_incam = self.smplx(**smpl_params_incam)
        verts_cam = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smplx_out_incam.vertices])
        output_dict['vertices_cam'] = verts_cam.cpu().numpy()
        
        # Process global coordinates (Centering & Grounding)
        if run_smpl24:
            joints_24_glob = torch.matmul(self.J_regressor_24, verts_glob).cpu().numpy()
            # Pelvis index is 0
            pelvis_x = joints_24_glob[0, 0, 0]
            pelvis_z = joints_24_glob[0, 0, 2]
            joints_24_glob[..., 0] -= pelvis_x
            joints_24_glob[..., 2] -= pelvis_z
            if rebase:
                min_y = joints_24_glob[0, :, 1].min()
                joints_24_glob[..., 1] -= min_y
            output_dict['smpl24'] = joints_24_glob
            
        if run_body25:
            joints_25_glob = torch.matmul(self.J_regressor_25, verts_glob).cpu().numpy()
            # MidHip index is 8
            pelvis_x = joints_25_glob[0, 8, 0]
            pelvis_z = joints_25_glob[0, 8, 2]
            joints_25_glob[..., 0] -= pelvis_x
            joints_25_glob[..., 2] -= pelvis_z
            if rebase:
                min_y = joints_25_glob[0, :, 1].min()
                joints_25_glob[..., 1] -= min_y
            output_dict['body25'] = joints_25_glob
            
        if run_h36m17:
            joints_17_glob = torch.matmul(self.J_regressor_17, verts_glob).cpu().numpy()
            # Pelvis index is 0
            pelvis_x = joints_17_glob[0, 0, 0]
            pelvis_z = joints_17_glob[0, 0, 2]
            joints_17_glob[..., 0] -= pelvis_x
            joints_17_glob[..., 2] -= pelvis_z
            if rebase:
                min_y = joints_17_glob[0, :, 1].min()
                joints_17_glob[..., 1] -= min_y
            output_dict['h36m17'] = joints_17_glob
            
        # Process camera space coordinates (for 2D overlays)
        joints_24_cam = torch.matmul(self.J_regressor_24, verts_cam).cpu().numpy()
        joints_25_cam = torch.matmul(self.J_regressor_25, verts_cam).cpu().numpy()
        joints_17_cam = torch.matmul(self.J_regressor_17, verts_cam).cpu().numpy()
        
        output_dict['joints_24_cam'] = joints_24_cam
        output_dict['joints_25_cam'] = joints_25_cam
        output_dict['joints_17_cam'] = joints_17_cam
        
        return output_dict


def main():
    parser = argparse.ArgumentParser(description="GVHMR SMPL24, BODY25, and H36M17 Detector Wrapper")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output-smpl24", help="Path to output SMPL24 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-body25", help="Path to output BODY25 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-h36m17", help="Path to output H36M17 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--device", default="cuda:0", help="GPU device index (default: cuda:0)")
    parser.add_argument("--disable-rebase", action="store_true", help="Disable rebasing lowest keypoint height to 0")
    parser.add_argument("--format", choices=['smpl24', 'body25', 'h36m17', 'all'], default='all', help="Output skeleton format (default: all)")
    parser.add_argument("--no-smpl24", action="store_true", help="Do not generate SMPL24 coordinates")
    parser.add_argument("--no-body25", action="store_true", help="Do not generate BODY25 coordinates")
    parser.add_argument("--static-cam", action="store_true", default=True, help="Assume static camera (default: True)")
    parser.add_argument("--use-slam", action="store_true", help="Use visual odometry to compute camera movement trajectory")
    parser.add_argument("--vis", action="store_true", help="Automatically run 3D skeleton joint visualizer scripts")
    parser.add_argument("--vis-skeleton-video", action="store_true", help="Visualize 2D projected skeleton overlaid back onto the video")
    parser.add_argument("--vis-smpl-video", action="store_true", help="Visualize 3D SMPL mesh overlaid back onto the video")
    
    args = parser.parse_args()
    
    run_smpl24 = args.format in ['smpl24', 'all']
    run_body25 = args.format in ['body25', 'all']
    run_h36m17 = args.format in ['h36m17', 'all']
    if args.no_smpl24:
        run_smpl24 = False
    if args.no_body25:
        run_body25 = False
        
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    output_smpl24 = args.output_smpl24
    if not output_smpl24 and run_smpl24:
        dir_smpl24 = os.path.join(project_root, "data", "smpl_joints", "samples_24j")
        os.makedirs(dir_smpl24, exist_ok=True)
        output_smpl24 = os.path.join(dir_smpl24, f"{base_name}_gvhmr_24j.npy")
        
    output_body25 = args.output_body25
    if not output_body25 and run_body25:
        dir_body25 = os.path.join(project_root, "data", "body25")
        os.makedirs(dir_body25, exist_ok=True)
        output_body25 = os.path.join(dir_body25, f"{base_name}_gvhmr_body25.npy")
        
    output_h36m17 = args.output_h36m17
    if not output_h36m17 and run_h36m17:
        dir_h36m17 = os.path.join(project_root, "data", "h36m17")
        os.makedirs(dir_h36m17, exist_ok=True)
        output_h36m17 = os.path.join(dir_h36m17, f"{base_name}_gvhmr_17j.npy")
        
    # Run Detector
    detector = GVHMRDetector(device=args.device)
    
    try:
        results = detector.detect_video(
            video_path=args.input,
            static_cam=not args.use_slam if args.static_cam else False,
            rebase=not args.disable_rebase,
            run_smpl24=run_smpl24,
            run_body25=run_body25,
            run_h36m17=run_h36m17
        )
        
        # Save results
        if run_smpl24 and results['smpl24'] is not None:
            os.makedirs(os.path.dirname(output_smpl24), exist_ok=True)
            np.save(output_smpl24, results['smpl24'].astype(np.float32))
            print(f"✅ SMPL24 output saved to: {output_smpl24} (Shape: {results['smpl24'].shape})")
            
        if run_body25 and results['body25'] is not None:
            os.makedirs(os.path.dirname(output_body25), exist_ok=True)
            np.save(output_body25, results['body25'].astype(np.float32))
            print(f"✅ BODY25 output saved to: {output_body25} (Shape: {results['body25'].shape})")
            
        if run_h36m17 and results['h36m17'] is not None:
            os.makedirs(os.path.dirname(output_h36m17), exist_ok=True)
            np.save(output_h36m17, results['h36m17'].astype(np.float32))
            print(f"✅ H36M17 output saved to: {output_h36m17} (Shape: {results['h36m17'].shape})")
            
        # 3D Visualizations
        if args.vis:
            if run_smpl24 and results['smpl24'] is not None:
                vis_script = os.path.join(project_root, "visualizers", "vis_smpl_joints.py")
                if os.path.exists(vis_script):
                    print(f"🎬 Visualizing SMPL24 using {os.path.basename(vis_script)}...")
                    subprocess.run([sys.executable, vis_script, output_smpl24])
                    
            if run_body25 and results['body25'] is not None:
                vis_script = os.path.join(project_root, "visualizers", "vis_body25_joints.py")
                if os.path.exists(vis_script):
                    print(f"🎬 Visualizing BODY25 using {os.path.basename(vis_script)}...")
                    subprocess.run([sys.executable, vis_script, output_body25])
                    
            if run_h36m17 and results['h36m17'] is not None:
                vis_script = os.path.join(project_root, "visualizers", "vis_smpl_joints.py")
                if os.path.exists(vis_script):
                    print(f"🎬 Visualizing H36M17 using {os.path.basename(vis_script)}...")
                    subprocess.run([sys.executable, vis_script, output_h36m17])
                    
        # 2D Skeleton Overlays
        if args.vis_skeleton_video:
            formats_to_vis = []
            if run_smpl24: formats_to_vis.append('smpl24')
            if run_body25: formats_to_vis.append('body25')
            if run_h36m17: formats_to_vis.append('h36m17')
            
            for fmt in formats_to_vis:
                if fmt == 'smpl24':
                    joints_cam_key = 'joints_24_cam'
                    dir_vis = os.path.join(project_root, "data", "smpl_joints", "samples_24j", "visualizations")
                elif fmt == 'h36m17':
                    joints_cam_key = 'joints_17_cam'
                    dir_vis = os.path.join(project_root, "data", "h36m17", "visualizations")
                else:
                    joints_cam_key = 'joints_25_cam'
                    dir_vis = os.path.join(project_root, "data", "body25", "visualizations")
                    
                joints_cam_val = results[joints_cam_key]
                if joints_cam_val is not None:
                    # Projection
                    joints_cam_t = torch.from_numpy(joints_cam_val)
                    K_t = torch.from_numpy(results['K_fullimg'])
                    joints_2d_t = perspective_projection(joints_cam_t, K_t)
                    joints_2d = joints_2d_t.numpy()
                    
                    os.makedirs(dir_vis, exist_ok=True)
                    overlay_path = os.path.join(dir_vis, f"overlay_skeleton_{base_name}_{fmt}_gvhmr.mp4")
                    vis_skeleton_video(
                        video_path=args.input,
                        joints_2d=joints_2d,
                        format_name=fmt,
                        output_video_path=overlay_path,
                        fps=30
                    )
                    print(f"✅ Skeleton overlay video saved to: {overlay_path}")
                    
        # 3D SMPL Mesh Overlay
        if args.vis_smpl_video:
            print("🎬 Rendering 3D SMPL mesh overlay video...")
            verts_cam = results['vertices_cam']
            faces = results['faces']
            K = results['K_fullimg']
            
            length, width, height = get_video_lwh(args.input)
            
            renderer = Renderer(width, height, device=args.device, faces=faces, K=torch.from_numpy(K[0]).to(args.device))
            reader = get_video_reader(args.input)
            
            dir_vis = os.path.join(project_root, "data", "smpl", "smpl", "visualizations")
            os.makedirs(dir_vis, exist_ok=True)
            overlay_path = os.path.join(dir_vis, f"overlay_smpl_{base_name}_gvhmr.mp4")
            
            writer = get_writer(overlay_path, fps=30, crf=23)
            for i, img_raw in enumerate(reader):
                img = renderer.render_mesh(torch.from_numpy(verts_cam[i]).to(args.device), img_raw, [0.8, 0.8, 0.8])
                writer.write_frame(img)
            writer.close()
            reader.close()
            
            # H.264 optimization using ffmpeg
            print(f"🔄 Optimizing video encoding (H.264)...")
            temp_path = overlay_path.replace(".mp4", "_temp.mp4")
            if os.path.exists(overlay_path):
                try:
                    os.rename(overlay_path, temp_path)
                    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", temp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", overlay_path]
                    subprocess.run(cmd, check=True)
                    os.remove(temp_path)
                except Exception as e:
                    print(f"⚠️ Video optimization failed: {e}")
                    if os.path.exists(temp_path):
                        os.rename(temp_path, overlay_path)
                        
            print(f"✅ SMPL mesh overlay video saved to: {overlay_path}")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
