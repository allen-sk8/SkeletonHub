"""
HybrIK 3D Skeleton Detector & Estimator (SMPL24 & OpenPose BODY25)

Example Usage:
    python detectors/hybrik_detector.py --input data/video.mp4 --vis

Technical Details & Reference Sources:
    - Reference Sources:
      * HybrIK Local Execution: `/home/allen/demo/0618boxing_demo/run_hybrik_local.py`
      * HybrIK Processor: `/home/leeyoyo49/Coachme_Service/api/engines/service-skeleton/processor.py`
      * HybrIK Service Fast: `/home/leeyoyo49/Coachme_Service/api/engines/service-skeleton/Hybrik/src/service_fast.py`
      * SMPL Parameter to BODY25: `converters/smpl_to_body_25j.py`
      * Coordinate & Scale Alignment: `converters/HybrIK_to_joints_24j.py` and `video2body25.sh`
    
    - Data Flow:
      1. Video Input ➔ Runs HybrIK model inside `hybrik_env` via subprocess.
      2. Outputs `skeleton.pk` (joints) and `smpl.pk` (SMPL params) in camera space.
      3. For SMPL24:
         - Extract `pred_xyz_24_struct_global` from `skeleton.pk`.
         - Flip Y and Z axes (Y-down camera to Y-up world).
         - Scale coordinates by 2.2 to restore physical meters.
         - Grounding: Offset Y so that the lowest joint of frame 0 is at Y = 0.
         - Save as `_hybrik_24j.npy`.
      4. For OpenPose BODY25:
         - Extract `pred_thetas`, `pred_betas`, and `transl` from `smpl.pk`.
         - Convert rotation matrices `pred_thetas` to axis-angle `poses`.
         - Run `SMPLHandler` forward kinematics in camera space to get 6890 mesh vertices.
         - Regress vertices to 25 joints using `J_regressor_body25.npy`.
         - Flip Y and Z axes (Y-down camera to Y-up world).
         - Centering: Center X and Z around the first frame's pelvis (MidHip, joint 8).
         - Grounding: Offset Y so that the lowest joint of frame 0 is at Y = 0.
         - Save as `_hybrik_body25.npy`.
"""
import os
import sys
import argparse
import pickle
import tempfile
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.smpl.handler import SMPLHandler

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
    (0, 255, 0),   # Spine -> Neck -> Nose -> REye -> REear (Center/Green)
    (0, 255, 0),   # Nose -> LEye -> LEar (Center/Green)
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

def project_camera_to_image(joints_cam, bboxes):
    """
    Project 3D joints in camera coordinates to 2D image coordinates.
    """
    T, J, _ = joints_cam.shape
    joints_2d = np.zeros((T, J, 2), dtype=np.float32)
    for t in range(T):
        x1, y1, x2, y2 = bboxes[t]
        w = x2 - x1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # In HybrIK, camera calibration/focal calculation internally assumes normalization by 256.0 height
        focal = 1000.0 * w / 256.0
        
        X = joints_cam[t, :, 0]
        Y = joints_cam[t, :, 1]
        Z = joints_cam[t, :, 2]
        
        Z = np.maximum(Z, 1e-5)
        joints_2d[t, :, 0] = (X * focal) / Z + cx
        joints_2d[t, :, 1] = (Y * focal) / Z + cy
    return joints_2d

def vis_skeleton_video(video_path, joints_2d, format_name, output_video_path, fps=20):
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
        fps = orig_fps if orig_fps > 0 else 20
        
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

def vis_smpl_video(video_path, vertices, faces, bboxes, output_video_path, fps=20):
    """
    Visualize 3D SMPL mesh overlaid back onto the input video.
    """
    import pyrender
    import trimesh
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = orig_fps if orig_fps > 0 else 20
        
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    renderer = pyrender.OffscreenRenderer(width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    t = 0
    total_frames = int(vertices.shape[0])
    
    print(f"🎬 Rendering SMPL mesh overlay video to: {output_video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret or t >= total_frames:
            break
            
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.4, 0.4, 0.4])
        
        mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, roughnessFactor=0.8,
            baseColorFactor=[0.2, 0.5, 0.8, 1.0]
        )
        mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh_node)
        
        x1, y1, x2, y2 = bboxes[t]
        w = x2 - x1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # In HybrIK, camera calibration/focal calculation internally assumes normalization by 256.0 height
        focal = 1000.0 * w / 256.0
        
        camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy)
        camera_pose = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        scene.add(camera, pose=camera_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        scene.add(light, pose=camera_pose)
        
        color, depth = renderer.render(scene)
        
        mask = depth > 0
        mesh_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        
        alpha = 0.7
        blended_frame = frame.copy()
        blended_frame[mask] = (alpha * mesh_bgr[mask] + (1.0 - alpha) * frame[mask]).astype(np.uint8)
        
        out.write(blended_frame)
        t += 1
        
    cap.release()
    out.release()
    renderer.delete()
    
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

class HybrikDetector:
    def __init__(self, device='cuda:0'):
        """
        Initialize the HybrIK detector interface.
        """
        self.device = device
        
        # Parse GPU index
        if "cuda" in device.lower():
            parts = device.split(':')
            self.gpu_id = int(parts[1]) if len(parts) > 1 else 0
        elif device.isdigit():
            self.gpu_id = int(device)
        else:
            self.gpu_id = 0
            
        print(f"⚙️ Initializing HybrikServiceFast on GPU {self.gpu_id}...")
        from utils.hybrik.service_fast import HybrikServiceFast
        self.service = HybrikServiceFast(gpu=self.gpu_id, skip_detection=True)

    def detect_video(self, video_path, flip_test=True, scale=2.2, rebase=True, run_smpl24=True, run_body25=True, run_h36m17=True, need_overlay=False):
        """
        Run HybrIK inference on a video and return estimated skeletons.
        
        Returns:
            dict containing processed outputs and raw camera space coordinates.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Input video file not found: {video_path}")

        from utils.hybrik.service_fast import HybrikFastRequest

        # 1. Setup temporary directory for intermediate outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_skeleton_pk = os.path.join(temp_dir, "skeleton.pk")
            temp_smpl_pk = os.path.join(temp_dir, "smpl.pk")
            
            print(f"⚙️ Running HybrIK inference natively in 'skeleton_env'...")
            
            run_reconstruct = run_body25 or need_overlay
            request = HybrikFastRequest(
                video_path=os.path.abspath(video_path),
                output_dir=temp_dir,
                save_name="skeleton",
                gpu=self.gpu_id,
                flip_test=flip_test,
                skip_detection=True,
                batch_size=32,
                save_vertices=run_reconstruct,
            )
            
            # Run inference directly inside python process
            self.service.process_video(request)
            
            if not os.path.exists(temp_skeleton_pk) or not os.path.exists(temp_smpl_pk):
                raise FileNotFoundError(
                    f"❌ Intermediate files were not generated by HybrIK."
                )

            # 2. Process outputs
            output_dict = {
                'smpl24': None,
                'body25': None,
                'h36m17': None,
                'joints_24_cam': None,
                'joints_25_cam': None,
                'joints_17_cam': None,
                'vertices': None,
                'faces': None,
                'bboxes': None
            }
            
            with open(temp_smpl_pk, 'rb') as f:
                smpl_data = pickle.load(f)
                
            pred_thetas = smpl_data['pred_thetas']
            pred_betas = smpl_data['pred_betas']
            transl = smpl_data['transl']
            bboxes = smpl_data['bbox']
            
            T = pred_thetas.shape[0]
            output_dict['bboxes'] = bboxes
            
            # Convert rotation matrices (T, 24, 3, 3) to axis-angle (T, 72)
            R_mats = pred_thetas.reshape(-1, 24, 3, 3)
            R_mats_flat = R_mats.reshape(-1, 3, 3)
            rot_vecs = R.from_matrix(R_mats_flat).as_rotvec()
            poses = rot_vecs.reshape(T, 72)
            
            handler = SMPLHandler(model_type='smpl')
            
            # --- Option A: Generate SMPL24 Joint Coordinates ---
            if run_smpl24:
                print("📐 Extracting SMPL24 Joint Coordinates...")
                with open(temp_skeleton_pk, 'rb') as f:
                    skel_data = pickle.load(f)
                    
                if 'pred_xyz_24_struct_global' in skel_data:
                    joints_24 = skel_data['pred_xyz_24_struct_global']
                else:
                    joints_24 = skel_data['features']
                    
                joints_24 = np.array(joints_24, dtype=np.float32)
                
                # Camera space (Y-down, Z-forward) -> World space (Y-up, Z-backward)
                joints_24[..., 1] *= -1
                joints_24[..., 2] *= -1
                
                # Scale from [-1, 1] normalization back to physical meters (2.2m scale)
                joints_24 *= scale
                
                # Apply rebasing/grounding if enabled
                if rebase:
                    # Grounding: Offset Y so the lowest joint of frame 0 is at Y = 0
                    min_y = float(joints_24[0, :, 1].min())
                    joints_24[..., 1] -= min_y
                    
                output_dict['smpl24'] = joints_24
                
            # Reconstruct camera-space SMPL24 joints if overlay is needed
            if need_overlay:
                # Use model's native camera-space predicted joints (pred_xyz_29 is pelvis-relative, so we scale by scale and add transl)
                pred_xyz_29 = smpl_data['pred_xyz_29']
                joints_24_cam = pred_xyz_29[:, :24, :] * scale + transl[:, None, :]
                output_dict['joints_24_cam'] = joints_24_cam
                
            # --- Option B: Generate OpenPose BODY25 Joint Coordinates ---
            if run_reconstruct:
                print("📐 Reconstructing OpenPose BODY25 coordinates from native model outputs...")
                # Get the model's native pelvis-relative pred_vertices and add transl to get camera-space vertices
                pred_vertices = smpl_data['pred_vertices']
                vertices = pred_vertices + transl[:, None, :]
                
                # Get faces from model template
                faces = handler._get_model('neutral').faces
                
                output_dict['vertices'] = vertices
                output_dict['faces'] = faces
                
                # Load J_regressor_body25
                regressor_path = os.path.join(project_root, 'common_models', 'regressor', 'J_regressor_body25.npy')
                if not os.path.exists(regressor_path):
                    fallback_path = os.path.join(project_root, 'external', 'EasyMocap', 'data', 'smplx', 'J_regressor_body25.npy')
                    if os.path.exists(fallback_path):
                        regressor_path = fallback_path
                    else:
                        raise FileNotFoundError(f"❌ Cannot find body25 regressor at {regressor_path}")
                        
                regressor = np.load(regressor_path)  # Shape: (25, 6890)
                
                # Regress vertices to body25 joints
                joints_25_cam = np.matmul(regressor[None], vertices)  # Shape: (T, 25, 3)
                output_dict['joints_25_cam'] = joints_25_cam
                
                if run_body25:
                    joints_25 = joints_25_cam.copy()
                    
                    # Camera space (Y-down, Z-forward) -> World space (Y-up, Z-backward)
                    joints_25[..., 1] *= -1
                    joints_25[..., 2] *= -1
                    
                    # Centering X and Z around first frame's pelvis (MidHip, joint 8)
                    orig_pelvis_x = float(joints_25[0, 8, 0])
                    orig_pelvis_z = float(joints_25[0, 8, 2])
                    joints_25[..., 0] -= orig_pelvis_x
                    joints_25[..., 2] -= orig_pelvis_z
                    
                    # Apply rebasing/grounding if enabled
                    if rebase:
                        # Grounding: Offset Y so the lowest joint of frame 0 is at Y = 0
                        min_y = float(joints_25[0, :, 1].min())
                        joints_25[..., 1] -= min_y
                        
                    output_dict['body25'] = joints_25
                    
            # --- Option C: Generate H36M-17 Joint Coordinates ---
            if run_h36m17 or need_overlay:
                pred_xyz_17 = smpl_data['pred_xyz_17']
                # Scale by scale and add transl to get camera-space coordinates in meters
                joints_17_cam = pred_xyz_17 * scale + transl[:, None, :]
                output_dict['joints_17_cam'] = joints_17_cam
                
                if run_h36m17:
                    print("📐 Extracting H36M-17 Joint Coordinates...")
                    joints_17 = joints_17_cam.copy()
                    
                    # Camera space (Y-down, Z-forward) -> World space (Y-up, Z-backward)
                    joints_17[..., 1] *= -1
                    joints_17[..., 2] *= -1
                    
                    # Centering X and Z around first frame's pelvis (Pelvis, joint 0)
                    orig_pelvis_x = float(joints_17[0, 0, 0])
                    orig_pelvis_z = float(joints_17[0, 0, 2])
                    joints_17[..., 0] -= orig_pelvis_x
                    joints_17[..., 2] -= orig_pelvis_z
                    
                    # Apply rebasing/grounding if enabled
                    if rebase:
                        # Grounding: Offset Y so the lowest joint of frame 0 is at Y = 0
                        min_y = float(joints_17[0, :, 1].min())
                        joints_17[..., 1] -= min_y
                        
                    output_dict['h36m17'] = joints_17
                    
            return output_dict

def main():
    parser = argparse.ArgumentParser(description="HybrIK SMPL24, BODY25, and H36M17 Detector Wrapper")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output-smpl24", help="Path to output SMPL24 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-body25", help="Path to output BODY25 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--output-h36m17", help="Path to output H36M17 .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--device", default="cuda:0", help="GPU device index (default: cuda:0)")
    parser.add_argument("--scale", type=float, default=2.2, help="Coordinates scale multiplier for smpl24/h36m17 (default: 2.2)")
    parser.add_argument("--disable-rebase", action="store_true", help="Disable rebasing lowest keypoint height to 0")
    parser.add_argument("--format", choices=['smpl24', 'body25', 'h36m17', 'all'], default='all', help="Output skeleton format (default: all)")
    parser.add_argument("--no-smpl24", action="store_true", help="Do not generate SMPL24 coordinates (backward compatible)")
    parser.add_argument("--no-body25", action="store_true", help="Do not generate BODY25 coordinates (backward compatible)")
    parser.add_argument("--vis", action="store_true", help="Automatically run 3D skeleton joint visualizer scripts")
    parser.add_argument("--vis-skeleton-video", action="store_true", help="Visualize 2D projected skeleton overlaid back onto the video")
    parser.add_argument("--vis-smpl-video", action="store_true", help="Visualize 3D SMPL mesh overlaid back onto the video")
    
    args = parser.parse_args()
    
    # Process format selections keeping backward compatibility
    run_smpl24 = args.format in ['smpl24', 'all']
    run_body25 = args.format in ['body25', 'all']
    run_h36m17 = args.format in ['h36m17', 'all']
    if args.no_smpl24:
        run_smpl24 = False
    if args.no_body25:
        run_body25 = False
        
    # 1. Setup auto-suffixing output paths
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    output_smpl24 = args.output_smpl24
    if not output_smpl24 and run_smpl24:
        dir_smpl24 = os.path.join(project_root, "data", "smpl_joints", "samples_24j")
        os.makedirs(dir_smpl24, exist_ok=True)
        output_smpl24 = os.path.join(dir_smpl24, f"{base_name}_hybrik_24j.npy")
        
    output_body25 = args.output_body25
    if not output_body25 and run_body25:
        dir_body25 = os.path.join(project_root, "data", "body25")
        os.makedirs(dir_body25, exist_ok=True)
        output_body25 = os.path.join(dir_body25, f"{base_name}_hybrik_body25.npy")
        
    output_h36m17 = args.output_h36m17
    if not output_h36m17 and run_h36m17:
        dir_h36m17 = os.path.join(project_root, "data", "h36m17")
        os.makedirs(dir_h36m17, exist_ok=True)
        output_h36m17 = os.path.join(dir_h36m17, f"{base_name}_hybrik_17j.npy")

    # 2. Run Detector
    detector = HybrikDetector(device=args.device)
    try:
        need_overlay = args.vis_skeleton_video or args.vis_smpl_video
        results = detector.detect_video(
            video_path=args.input,
            flip_test=True,
            scale=args.scale,
            rebase=not args.disable_rebase,
            run_smpl24=run_smpl24,
            run_body25=run_body25,
            run_h36m17=run_h36m17,
            need_overlay=need_overlay
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
            
        # 3. Optional 3D Visualizations
        if args.vis:
            if run_smpl24 and results['smpl24'] is not None:
                vis_script = os.path.join(project_root, "visualizers", "vis_smpl_joints.py")
                if os.path.exists(vis_script):
                    print(f"🎬 Visualizing SMPL24 using {os.path.basename(vis_script)}...")
                    subprocess.run([sys.executable, vis_script, output_smpl24])
                else:
                    print(f"⚠️ Visualizer not found: {vis_script}")
                    
            if run_body25 and results['body25'] is not None:
                vis_script = os.path.join(project_root, "visualizers", "vis_body25_joints.py")
                if os.path.exists(vis_script):
                    print(f"🎬 Visualizing BODY25 using {os.path.basename(vis_script)}...")
                    subprocess.run([sys.executable, vis_script, output_body25])
                else:
                    print(f"⚠️ Visualizer not found: {vis_script}")
                    
            if run_h36m17 and results['h36m17'] is not None:
                vis_script = os.path.join(project_root, "visualizers", "vis_smpl_joints.py")
                if os.path.exists(vis_script):
                    print(f"🎬 Visualizing H36M17 using {os.path.basename(vis_script)}...")
                    subprocess.run([sys.executable, vis_script, output_h36m17])
                else:
                    print(f"⚠️ Visualizer not found: {vis_script}")
                    
        # 4. Optional Overlay Visualizations (Mapped back onto video)
        if args.vis_skeleton_video:
            formats_to_vis = []
            if run_smpl24: formats_to_vis.append('smpl24')
            if run_body25: formats_to_vis.append('body25')
            if run_h36m17: formats_to_vis.append('h36m17')
            
            for fmt in formats_to_vis:
                if fmt == 'smpl24':
                    joints_cam_key = 'joints_24_cam'
                elif fmt == 'h36m17':
                    joints_cam_key = 'joints_17_cam'
                else:
                    joints_cam_key = 'joints_25_cam'
                    
                joints_cam = results[joints_cam_key]
                if joints_cam is not None:
                    joints_2d = project_camera_to_image(joints_cam, results['bboxes'])
                    
                    if fmt == 'smpl24':
                        dir_vis = os.path.join(project_root, "data", "smpl_joints", "samples_24j", "visualizations")
                    elif fmt == 'h36m17':
                        dir_vis = os.path.join(project_root, "data", "h36m17", "visualizations")
                    else:
                        dir_vis = os.path.join(project_root, "data", "body25", "visualizations")
                    os.makedirs(dir_vis, exist_ok=True)
                    
                    overlay_path = os.path.join(dir_vis, f"overlay_skeleton_{base_name}.mp4")
                    vis_skeleton_video(
                        video_path=args.input,
                        joints_2d=joints_2d,
                        format_name=fmt,
                        output_video_path=overlay_path,
                        fps=20
                    )
                    print(f"✅ Skeleton overlay video saved to: {overlay_path}")
                    
        if args.vis_smpl_video:
            if results['vertices'] is not None and results['faces'] is not None:
                dir_vis = os.path.join(project_root, "data", "smpl", "smpl", "visualizations")
                os.makedirs(dir_vis, exist_ok=True)
                
                overlay_path = os.path.join(dir_vis, f"overlay_smpl_{base_name}.mp4")
                vis_smpl_video(
                    video_path=args.input,
                    vertices=results['vertices'],
                    faces=results['faces'],
                    bboxes=results['bboxes'],
                    output_video_path=overlay_path,
                    fps=20
                )
                print(f"✅ SMPL mesh overlay video saved to: {overlay_path}")
                    
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
