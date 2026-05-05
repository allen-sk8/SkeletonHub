"""
SMPL Inverse Kinematics Solver (Batch)
從 24 關節 XYZ 座標反推 SMPL 參數 (poses, betas, trans)

核心設計：
- 直接使用 smplx 進行前向運動學 (Forward Kinematics)，不依賴 EasyMocap
- 全序列批次優化 (Batch Optimization)，大幅加速
- 四階段優化策略：
  Stage 1: Shape 先擬合骨長
  Stage 2: 全域旋轉 + 平移 (Global Orient + Translation)
  Stage 3: 全身體姿態 (Body Pose) + 全域參數 + 時序平滑
  Stage 4: 純關節位置精修 (不含 smooth，最大化精度)
- 無死鎖關節：所有 24 個關節都參與優化

座標約定：輸入必須為 Y-up, 公尺 (meters)
"""

import os
import sys
import torch
import numpy as np
import smplx

# SMPL 24 joints 的 kinematic chain (parent → child)
SMPL_KINEMATIC_PAIRS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]


class SMPLIKSolver:
    """
    SMPL Inverse Kinematics Solver (Batch)
    從 3D joint positions 反推 SMPL pose parameters
    """
    
    def __init__(self, model_root="common_models/body_models", gender="neutral", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gender = gender
        
        # 載入 SMPL 模型（batch_size 之後動態設定）
        self.model_root = model_root
        self.model = None
        self._load_model(gender, batch_size=1)
    
    def _load_model(self, gender, batch_size=1):
        """載入或重新載入 SMPL 模型"""
        for g in [gender, 'female', 'male', 'neutral']:
            try:
                self.model = smplx.create(
                    model_path=self.model_root,
                    model_type='smpl',
                    gender=g,
                    batch_size=batch_size,
                    create_transl=True,
                ).to(self.device)
                self.gender = g
                print(f"✅ SMPL 模型已載入 (性別: {g}, batch_size: {batch_size})")
                return
            except:
                continue
        raise FileNotFoundError(f"❌ 無法在 {self.model_root} 找到 SMPL 模型")
    
    def _forward_batch(self, global_orient, body_pose, betas, transl):
        """批次 Forward Kinematics"""
        T = global_orient.shape[0]
        
        # 如果 model batch_size 不匹配，重新載入
        if self.model.batch_size != T:
            self._load_model(self.gender, batch_size=T)
        
        # betas 可能是 (1, 10)，需要 expand 到 (T, 10)
        if betas.shape[0] == 1 and T > 1:
            betas_expanded = betas.expand(T, -1)
        else:
            betas_expanded = betas
        
        output = self.model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_expanded,
            transl=transl,
            return_verts=False,
        )
        return output.joints[:, :24, :]  # (T, 24, 3)
    
    def _bone_length_loss(self, pred, target):
        """骨長一致性 loss"""
        loss = torch.tensor(0.0, device=self.device)
        for (p, c) in SMPL_KINEMATIC_PAIRS:
            pred_len = torch.norm(pred[:, p] - pred[:, c], dim=-1)
            tgt_len = torch.norm(target[:, p] - target[:, c], dim=-1)
            loss += torch.mean((pred_len - tgt_len) ** 2)
        return loss
    
    def _anatomical_loss(self, body_pose):
        """
        解剖學約束 loss - 解決 twist ambiguity（扭轉歧義）
        
        SMPL body_pose 是 (T, 69)，每 3 個值 = 1 個關節的 axis-angle。
        body_pose 關節索引 (0-based in body_pose, 1-based in SMPL):
          0=L_Hip, 1=R_Hip, 2=Spine1, 3=L_Knee, 4=R_Knee, 5=Spine2,
          6=L_Ankle, 7=R_Ankle, 8=Spine3, ...
        
        核心約束：
        - 膝蓋主要繞 X 軸彎曲，Y/Z 軸旋轉應極小 (防止扭腿)
        - 肘部同理
        - 脊椎的 twist (Y 軸) 應受限
        """
        loss = torch.tensor(0.0, device=self.device)
        
        # 膝蓋: body_pose indices 3(L_Knee), 4(R_Knee)
        # 膝蓋幾乎只能繞 X 軸彎曲 (flex/extend)，Y/Z 旋轉幾乎為零
        for knee_idx in [3, 4]:  # L_Knee, R_Knee
            knee_rot = body_pose[:, knee_idx*3:(knee_idx+1)*3]
            # Y, Z 分量應趨近零
            loss += torch.mean(knee_rot[:, 1] ** 2) * 5.0  # Y (twist)
            loss += torch.mean(knee_rot[:, 2] ** 2) * 5.0  # Z (lateral)
            # X 應該是正值 (膝蓋只能向前彎)
            loss += torch.mean(torch.relu(-knee_rot[:, 0])) * 1.0
        
        # 肘部: body_pose indices 15(L_Elbow=18-3=15), 16(R_Elbow=19-3=16)  
        for elbow_idx in [15, 16]:  # L_Elbow, R_Elbow
            elbow_rot = body_pose[:, elbow_idx*3:(elbow_idx+1)*3]
            loss += torch.mean(elbow_rot[:, 1] ** 2) * 2.0
            loss += torch.mean(elbow_rot[:, 2] ** 2) * 2.0
        
        # 腳踝: body_pose indices 6(L_Ankle), 7(R_Ankle) - 限制扭轉
        for ankle_idx in [6, 7]:
            ankle_rot = body_pose[:, ankle_idx*3:(ankle_idx+1)*3]
            loss += torch.mean(ankle_rot[:, 1] ** 2) * 2.0  # Y twist
        
        return loss
    
    def fit_sequence(self, target_joints_seq, verbose=True):
        """
        對整個動作序列進行批次 IK 求解
        
        Args:
            target_joints_seq: (T, 24, 3) numpy array, Y-up, meters
            verbose: 是否印出進度
        
        Returns:
            dict with 'poses' (T, 72), 'betas' (1, 10), 'trans' (T, 3), 'gender'
        """
        T = target_joints_seq.shape[0]
        target = torch.tensor(target_joints_seq, dtype=torch.float32, device=self.device)  # (T, 24, 3)
        
        # 初始化參數
        global_orient = torch.zeros(T, 3, device=self.device, requires_grad=True)
        body_pose = torch.zeros(T, 69, device=self.device, requires_grad=True)
        betas = torch.zeros(1, 10, device=self.device, requires_grad=True)
        # Translation 用 pelvis 位置初始化
        transl = torch.tensor(target_joints_seq[:, 0, :].copy(), dtype=torch.float32, 
                              device=self.device, requires_grad=True)
        
        if verbose:
            print(f"🚀 開始批次 IK 求解: {T} 幀 (裝置: {self.device})")
        
        # === Stage 1: Shape (骨長擬合) ===
        if verbose:
            print(f"  📐 Stage 1: Shape 骨長擬合 (200 iters)...")
        optimizer = torch.optim.Adam([betas], lr=0.01)
        for i in range(200):
            optimizer.zero_grad()
            pred = self._forward_batch(global_orient.detach(), body_pose.detach(), betas, transl.detach())
            loss = self._bone_length_loss(pred, target)
            loss.backward()
            optimizer.step()
            if verbose and (i + 1) % 100 == 0:
                print(f"    iter {i+1}: bone_loss={loss.item():.8f}")
        
        # === Stage 2: Global Orient + Translation ===
        if verbose:
            print(f"  📐 Stage 2: Global Orient + Translation (300 iters)...")
        optimizer = torch.optim.Adam([global_orient, transl], lr=0.02)
        for i in range(300):
            optimizer.zero_grad()
            pred = self._forward_batch(global_orient, body_pose.detach(), betas.detach(), transl)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            if verbose and (i + 1) % 100 == 0:
                print(f"    iter {i+1}: loss={loss.item():.6f}")
        
        # === Stage 3: Full Pose + Smooth + Anatomical ===
        if verbose:
            print(f"  📐 Stage 3: Full Pose + Anatomical Constraints (800 iters)...")
        optimizer = torch.optim.Adam([global_orient, body_pose, transl], lr=0.01)
        for i in range(800):
            optimizer.zero_grad()
            pred = self._forward_batch(global_orient, body_pose, betas.detach(), transl)
            
            loss_joint = torch.mean((pred - target) ** 2)
            
            # 時序平滑 (權重遞減)
            loss_smooth = torch.tensor(0.0, device=self.device)
            if T > 1:
                w_smooth = max(0.05 * (1 - i / 800), 0.005)
                loss_smooth += torch.mean((body_pose[1:] - body_pose[:-1]) ** 2) * w_smooth
                loss_smooth += torch.mean((global_orient[1:] - global_orient[:-1]) ** 2) * w_smooth
            
            # 解剖學約束 (權重遞減，讓精修階段不被約束限制)
            w_anat = max(0.01 * (1 - i / 800), 0.001)
            loss_anat = self._anatomical_loss(body_pose) * w_anat
            
            # 極輕的正則化
            loss_reg = torch.mean(body_pose ** 2) * 0.0001
            
            loss = loss_joint + loss_smooth + loss_anat + loss_reg
            loss.backward()
            optimizer.step()
            if verbose and (i + 1) % 200 == 0:
                print(f"    iter {i+1}: total={loss.item():.6f} joint={loss_joint.item():.6f} anat={loss_anat.item():.6f}")
        
        # === Stage 4: 純精修 (只有 joint + 輕解剖學) ===
        if verbose:
            print(f"  📐 Stage 4: 精修 + 輕解剖學約束 (500 iters)...")
        optimizer = torch.optim.Adam([global_orient, body_pose, betas, transl], lr=0.003)
        for i in range(500):
            optimizer.zero_grad()
            pred = self._forward_batch(global_orient, body_pose, betas, transl)
            loss_joint = torch.mean((pred - target) ** 2)
            loss_anat = self._anatomical_loss(body_pose) * 0.0005
            loss = loss_joint + loss_anat
            loss.backward()
            optimizer.step()
            if verbose and (i + 1) % 100 == 0:
                print(f"    iter {i+1}: joint={loss_joint.item():.8f} anat={loss_anat.item():.8f}")
        
        # 組裝結果
        poses = torch.cat([global_orient, body_pose], dim=1).detach().cpu().numpy()  # (T, 72)
        trans = transl.detach().cpu().numpy()  # (T, 3)
        betas_np = betas.detach().cpu().numpy()  # (1, 10)
        
        if verbose:
            # 計算最終誤差
            with torch.no_grad():
                pred_final = self._forward_batch(global_orient, body_pose, betas, transl)
                per_joint_err = torch.norm(pred_final - target, dim=-1)  # (T, 24)
                mean_err = torch.mean(per_joint_err).item()
                max_err = torch.max(per_joint_err).item()
                # 找出最大誤差的幀和關節
                max_idx = torch.argmax(per_joint_err)
                max_frame = (max_idx // 24).item()
                max_joint = (max_idx % 24).item()
            print(f"✅ IK 求解完成！")
            print(f"   平均關節誤差: {mean_err*100:.2f} cm")
            print(f"   最大誤差: {max_err*100:.2f} cm (Frame {max_frame}, Joint {max_joint})")
        
        return {
            'poses': poses.astype(np.float32),
            'betas': betas_np.astype(np.float32),
            'trans': trans.astype(np.float32),
            'gender': self.gender,
            'mocap_framerate': 20.0
        }
