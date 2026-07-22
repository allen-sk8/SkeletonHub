import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# Add project root and rtmpose3d directories to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

rtmpose3d_path = os.path.join(project_root, 'external', 'mmpose', 'projects', 'rtmpose3d')
sys.path.insert(0, rtmpose3d_path)

# Import axis converter from utility
from utils.axis_converter import convert_joints_z_to_y

# Import mmpose APIs (will work when run inside skeleton_env conda environment)
try:
    from mmpose.apis import inference_topdown, init_model
    from mmdet.apis import inference_detector, init_detector
    from mmpose.utils import adapt_mmdet_pipeline
    from rtmpose3d import *  # Register rtmpose3d models/heads
    HAS_MMPOSE = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_MMPOSE = False
    MMPOSE_ERROR = e

class RTMPose3DDetector:
    def __init__(self, det_config=None, det_checkpoint=None, pose_config=None, pose_checkpoint=None, device='cuda:0'):
        if not HAS_MMPOSE:
            raise ImportError(f"Cannot import mmpose/mmdet components. Please run within 'skeleton_env' environment.\nDetail: {MMPOSE_ERROR}")
        
        # Set default configurations if not specified
        if det_config is None:
            det_config = os.path.join(rtmpose3d_path, 'demo', 'rtmdet_m_640-8xb32_coco-person.py')
        if det_checkpoint is None:
            det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
            
        if pose_config is None:
            pose_config = os.path.join(rtmpose3d_path, 'configs', 'rtmw3d-x_8xb32_cocktail14-384x288.py')
        if pose_checkpoint is None:
            pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth'

        print(f"⚙️ Initializing Detector: {os.path.basename(det_config)}")
        self.detector = init_detector(det_config, det_checkpoint, device=device.lower())
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        
        print(f"⚙️ Initializing Pose Estimator: {os.path.basename(pose_config)}")
        self.pose_estimator = init_model(pose_config, pose_checkpoint, device=device.lower())
        
        self.device = device
        
    def detect_video(self, video_path, rebase=True, bbox_thr=0.5, bbox_file=None):
        """
        Run 3D pose estimation on a video and return a numpy array of shape (T, 133, 3) in Y-up meters.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Processing video: {video_path} ({total_frames} frames)")
        
        precomputed_bboxes = None
        if bbox_file is not None and os.path.exists(bbox_file):
            print(f"📦 Loading precomputed bounding boxes from: {bbox_file}")
            precomputed_bboxes = []
            with open(bbox_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        precomputed_bboxes.append([float(p) for p in parts])
            precomputed_bboxes = np.array(precomputed_bboxes, dtype=np.float32)

        frames_joints = []
        last_valid_joints = np.zeros((133, 3), dtype=np.float32)
        
        # Iterate over video frames with tqdm
        for frame_idx in tqdm(range(total_frames), desc="Running Pose Inference"):
            ret, frame = cap.read()
            if not ret:
                break
                
            main_bbox = None
            if precomputed_bboxes is not None and frame_idx < len(precomputed_bboxes):
                box = precomputed_bboxes[frame_idx]
                if not (box[0] == -1.0 and box[1] == -1.0):
                    main_bbox = np.array([box], dtype=np.float32)

            if main_bbox is None:
                # 1. Run Person Detector
                det_result = inference_detector(self.detector, frame)
                pred_instances = det_result.pred_instances.cpu().numpy()
                
                # Filter person bounding boxes
                bboxes = pred_instances.bboxes
                scores = pred_instances.scores
                labels = pred_instances.labels
                
                # Select person class (0 in COCO) above threshold
                valid_idx = np.logical_and(labels == 0, scores > bbox_thr)
                bboxes = bboxes[valid_idx]
                scores = scores[valid_idx]
                
                if len(bboxes) == 0:
                    # No person detected, carry over the last frame's joints
                    frames_joints.append(last_valid_joints.copy())
                    continue
                    
                # Heuristic: Select the main person (largest bounding box area)
                bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                main_person_idx = np.argmax(bbox_areas)
                main_bbox = bboxes[main_person_idx : main_person_idx + 1]
                
            # 2. Run 3D Pose Estimator
            pose_est_results = inference_topdown(self.pose_estimator, frame, main_bbox)
            
            if len(pose_est_results) == 0:
                frames_joints.append(last_valid_joints.copy())
                continue
                
            pred_instances_pose = pose_est_results[0].pred_instances
            keypoints = pred_instances_pose.keypoints  # Shape: (1, 133, 3) or (133, 3)
            
            # Squeeze to (133, 3)
            if keypoints.ndim == 3:
                keypoints = np.squeeze(keypoints, axis=0)
            elif keypoints.ndim == 4:
                keypoints = np.squeeze(np.squeeze(keypoints, axis=0), axis=0)
                
            # 3. Coordinate Transformation (Following body3d_img2pose_demo.py logic)
            # Swap: keypoints = -keypoints[..., [0, 2, 1]]
            keypoints_demo = -keypoints[..., [0, 2, 1]]
            
            if rebase:
                keypoints_demo[..., 2] -= np.min(keypoints_demo[..., 2], axis=-1, keepdims=True)
                
            # 4. Convert from Z-up to Y-up (Project Standard)
            # Note: Keypoints are already estimated in meters by the model
            keypoints_y_up = convert_joints_z_to_y(keypoints_demo)
            
            frames_joints.append(keypoints_y_up.copy())
            last_valid_joints = keypoints_y_up
            
        cap.release()
        
        # Stack into (T, 133, 3)
        if len(frames_joints) == 0:
            return np.zeros((0, 133, 3), dtype=np.float32)
            
        return np.stack(frames_joints, axis=0)

def main():
    parser = argparse.ArgumentParser(description="RTMPose3D 133-joint Detector")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", help="Path to output .npy file. If omitted, uses auto-suffixing.")
    parser.add_argument("--device", default="cuda:0", help="GPU device index (default: cuda:0)")
    parser.add_argument("--bbox-file", help="Path to precomputed 2D bounding boxes text file (optional)")
    parser.add_argument("--disable-rebase", action="store_true", help="Disable rebasing lowest keypoint height to 0")
    
    args = parser.parse_args()
    
    # Check environment
    if not HAS_MMPOSE:
        print(f"❌ Error: Cannot run. Please activate 'skeleton_env' conda environment.")
        print(f"Detail: {MMPOSE_ERROR}")
        sys.exit(1)
        
    # Setup auto-suffixing output path if not specified
    output_path = args.output
    if not output_path:
        base_dir = os.path.join(project_root, "data", "coco_wholebody133")
        os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(base_dir, f"{base_name}_coco_wholebody133.npy")
        
    # Run Detector
    try:
        detector = RTMPose3DDetector(device=args.device)
        joints_133 = detector.detect_video(
            video_path=args.input,
            rebase=not args.disable_rebase,
            bbox_file=args.bbox_file
        )
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints_133.astype(np.float32))
        print(f"✅ Detection complete. Output shape: {joints_133.shape} (T, 133, 3)")
        print(f"💾 Saved 133-joint sequence to: {output_path}")
        
    except Exception as e:
        print(f"❌ An error occurred during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
