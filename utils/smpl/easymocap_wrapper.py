import os
import torch
import numpy as np
import sys
from utils.axis_converter import convert_joints_y_to_z, convert_smpl_z_to_y

# Ensure EasyMocap can be imported
# (Though it should be installed in editable mode, adding to path as fallback)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/EasyMocap'))

try:
    from easymocap.bodymodel.smpl import SMPLModel
    from easymocap.pyfitting.optimize_simple import optimizePose3D
    from yacs.config import CfgNode as CN
except ImportError as e:
    print(f"❌ [EasyMocapWrapper] Import failed: {e}. Make sure EasyMocap is installed.")
    raise e

class SMPLModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_type = model.model_type
        self.device = model.device
        self.init_params = model.init_params
    
    def forward(self, *args, **kwargs):
        # Force return_smpl_joints=True to include the joints we want to fit
        kwargs['return_smpl_joints'] = True
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'model':
            return super().__getattr__(name)
        return getattr(self.model, name)

class EasyMocapWrapper:
    """
    A wrapper for EasyMocap to provide a clean API for SMPL fitting.
    Follows SkeletonHub standards: Y-up, meters, Right-handed.
    """
    def __init__(self, model_root="common_models/body_models", model_type="smplh", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Determine model path
        # Default to MALE if neutral is missing (standard for SMPL-H)
        if model_type == 'smplh':
            model_path = os.path.join(model_root, "smplh", "SMPLH_MALE.pkl")
        else:
            model_path = os.path.join(model_root, "smpl", "SMPL_NEUTRAL.pkl")
            
        if not os.path.exists(model_path):
            # Fallback check for other genders if neutral is missing
            for g in ['female', 'male']:
                path = model_path.replace('NEUTRAL', g.upper())
                if os.path.exists(path):
                    model_path = path
                    break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ [EasyMocapWrapper] SMPL model not found at {model_path}")
            
        print(f"🚀 [EasyMocapWrapper] Loading {model_type} from {model_path}")
        # Note: EasyMocap's SMPLModel handles its own loading
        raw_model = SMPLModel(model_path, device=self.device, model_type=model_type)
        self.body_model = SMPLModelWrapper(raw_model)
        
        # Default config for EasyMocap fitting
        self.cfg = self._get_default_config()

    def _get_default_config(self):
        cfg = CN()
        cfg.device = str(self.device)
        cfg.verbose = True
        cfg.print_freq = 10 # 提高打印頻率，充當進度條
        cfg.model = self.model_type
        cfg.OPT_R = True
        cfg.OPT_T = True
        cfg.OPT_POSE = True
        cfg.OPT_SHAPE = False 
        cfg.OPT_HAND = False  
        cfg.OPT_EXPR = False
        return cfg

    def fit_3d(self, target_joints, use_hand=False, num_iters=100):
        """
        Fit SMPL/SMPL-H to 3D joints.
        target_joints: (T, J, 3) in Y-up, meters.
        returns: dict with 'poses', 'betas', 'trans', 'gender'
        """
        T, J, _ = target_joints.shape
        if J < 25:
            # Pad to 25 joints to avoid IndexError in EasyMocap's LossRegPosesZero
            padding = np.zeros((T, 25 - J, 3))
            target_joints = np.concatenate([target_joints, padding], axis=1)
            J = 25
            
        self.cfg.OPT_HAND = use_hand and (self.model_type == 'smplh')
        
        # 1. Coordinate Transformation: Y-up -> Z-up (EasyMocap standard internal logic)
        # 使用統一的轉換工具
        joints_z_up = convert_joints_y_to_z(target_joints)
        
        # Add confidence score (all 1.0 for now)
        k3d = np.concatenate([joints_z_up, np.ones((T, J, 1))], axis=-1)
        
        print(f"📊 [EasyMocapWrapper] k3d mean: {k3d[..., :3].mean():.4f}, std: {k3d[..., :3].std():.4f}")
        print(f"📊 [EasyMocapWrapper] k3d sample (frame 0, joint 0): {k3d[0, 0, :3]}")

        # 2. Initialize Parameters
        params = self.body_model.init_params(nFrames=T)
        # Simple translation init based on root (usually joint 0)
        params['Th'] = joints_z_up[:, 0, :]
        
        # 3. Define Weights
        # 關鍵：k3d 必須遠大於所有正則項之和，否則模型會「抗拒擬合」
        weights = {
            'k3d': 500.0,            # 增加權重，確保緊貼
            'reg_poses': 5.0,        # 調高正則，防止「麻花捲」扭轉
            'smooth_body': 100.0,    
            'smooth_poses': 100.0,   # 加強平滑，減少幀間抖動
            'smooth_Rh': 20.0,       
            'init_poses': 1.0,       # 增加初始姿勢約束，防止姿勢跑飛
        }
        if self.cfg.OPT_HAND:
            weights['k3d_hand'] = 100.0
            weights['reg_hand'] = 5.0
            weights['smooth_hand'] = 50.0

        # 4. Run Optimization (Three-Stage for extreme stability)
        
        # --- Stage 0: Deep fit for the FIRST frame to set a good starting point ---
        print(f"🚀 [EasyMocapWrapper] Stage 0: Deep fitting first frame to avoid curling...")
        self.cfg.OPT_POSE = True
        self.cfg.OPT_R = True
        self.cfg.OPT_T = True
        self.cfg.n_iter = 100
        
        # Only fit frame 0
        k3d_f0 = k3d[0:1]
        params_f0 = self.body_model.init_params(nFrames=1)
        params_f0['Th'][0] = joints_z_up[0, 0, :]
        
        # High reg to ensure it's a clean pose
        weights_f0 = weights.copy()
        weights_f0['smooth_body'] = 0.0 # Single frame no smooth
        weights_f0['smooth_poses'] = 0.0
        weights_f0['smooth_Rh'] = 0.0
        weights_f0['reg_poses'] = 10.0 # 提高正則，確保第一幀絕對不扭轉
        
        fitted_f0 = optimizePose3D(self.body_model, params_f0, k3d_f0, weights_f0, self.cfg)
        
        # Use Stage 0 results to initialize ALL frames
        for key in ['Rh', 'Th', 'poses']:
            # Expand frame 0 params to all frames
            params[key] = np.repeat(fitted_f0[key], T, axis=0)
            
        # --- Stage 1: Global alignment for sequence ---
        print(f"🚀 [EasyMocapWrapper] Stage 1: Global sequence alignment...")
        self.cfg.OPT_POSE = False
        self.cfg.n_iter = 50
        weights_stage1 = weights.copy()
        weights_stage1['reg_poses'] = 0.0
        params_stage1 = optimizePose3D(self.body_model, params, k3d, weights_stage1, self.cfg)
        
        # --- Stage 2: Full Body Pose ---
        print(f"🚀 [EasyMocapWrapper] Stage 2: Final refinement (iters: {num_iters})...")
        self.cfg.OPT_POSE = True
        self.cfg.n_iter = num_iters
        fitted_params = optimizePose3D(self.body_model, params_stage1, k3d, weights, self.cfg)
        
        # 5. Convert Results back to SkeletonHub Standard
        # Poses: (Rh, body_pose, hand_pose)
        # EasyMocap returns Rh and Th in its internal (Z-up) space.
        # However, we want the parameters to be compatible with our SMPLHandler (which renders in Y-up).
        # Standard SMPL renders usually expect Z-up if no rotation is applied.
        
        final_poses = np.concatenate([fitted_params['Rh'], fitted_params['poses']], axis=-1)
        
        # Padding for SMPL-H if hands are missing
        if self.model_type == 'smplh' and final_poses.shape[-1] < 156:
            padding = np.zeros((T, 156 - final_poses.shape[-1]))
            final_poses = np.concatenate([final_poses, padding], axis=-1)

        # 5. Prepare result and Convert to Y-up (Project Standard)
        # Note: EasyMocap's optimizePose3D already returns numpy arrays
        result = {
            'poses': final_poses,
            'trans': fitted_params['Th'],
            'betas': fitted_params['shapes'] if 'shapes' in fitted_params else np.zeros(10),
            'gender': 'male' if self.model_type == 'smplh' else 'neutral',
            'mocap_framerate': 20.0 # Default
        }
        
        # 🌟 既然專案要求進入內部的東西都是 Y-up，我們在此進行轉換
        result = convert_smpl_z_to_y(result)
        
        return result
