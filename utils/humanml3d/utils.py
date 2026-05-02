import torch
import numpy as np
import os
import sys

from .lib.skeleton import Skeleton
from .lib.quaternion import qrot, qinv, quaternion_to_cont6d, qbetween_np, qrot_np, qinv_np, qfix, qmul_np, quaternion_to_cont6d_np
from .lib.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

# --- HumanML3D Constants ---
# Lower legs
L_IDX1, L_IDX2 = 5, 8
# Right/Left foot
FID_R, FID_L = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
FACE_JOINT_INDX = [2, 1, 17, 16]
# l_hip, r_hip
R_HIP, L_HIP = 2, 1
JOINTS_NUM = 22

def recover_root_rot_pos(data):
    """
    從 263D 特徵中恢復根節點的旋轉與位置
    data: (T, 263)
    """
    rot_vel = data[..., 0]
    r_rot_ang = np.zeros_like(rot_vel)
    # 從旋轉速度恢復 Y 軸旋轉角度
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)

    r_rot_quat = np.zeros(data.shape[:-1] + (4,))
    r_rot_quat[..., 0] = np.cos(r_rot_ang)
    r_rot_quat[..., 2] = np.sin(r_rot_ang)

    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    
    # 將 Y 軸旋轉應用到根節點位置 (XZ 平面速度轉為世界座標)
    r_pos = qrot_np(qinv_np(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[..., 1] = data[..., 3] # 根節點高度
    
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num=22):
    """
    從 HumanML3D 的 RIC (Local Joint Positions) 恢復全局關節座標
    data: (T, 263)
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # 提取局部關節位置 (RIC data starts from index 4)
    # indices: 4 to 4 + (21*3)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.reshape(positions.shape[:-1] + (-1, 3))

    # 1. 旋轉回到世界座標 (繞 Y 軸旋轉)
    positions = qrot_np(qinv_np(np.repeat(r_rot_quat[..., None, :], positions.shape[-2], axis=-2)), positions)

    # 2. 加入根節點的 XZ 平移
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # 3. 合併根節點與其他關節
    global_positions = np.concatenate([r_pos[..., np.newaxis, :], positions], axis=-2)

    return global_positions

def foot_detect(positions, thres):
    """
    偵測腳底接觸
    positions: (T, 22, 3)
    """
    velfactor = np.array([thres, thres])
    
    # Left foot (7, 10)
    feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
    feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
    feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    # Right foot (8, 11)
    feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
    feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
    feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
    
    return feet_l, feet_r

def get_cont6d_params(positions):
    """
    計算關節旋轉特徵 (6D)
    """
    skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
    # (T, 22, 4)
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX, smooth_forward=True)

    # Quaternion to continuous 6D
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()
    
    # Root Linear Velocity
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)
    
    # Root Angular Velocity
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    
    return cont_6d_params, r_velocity, velocity, r_rot

def extract_features(positions, feet_thre=0.002):
    """
    從 3D 關節座標提取 263D 特徵向量 (XYZ -> 263D)
    positions: (T, 22, 3)
    """
    seq_len = positions.shape[0]
    
    # 1. 偵測腳底接觸 (T-1, 4)
    feet_l, feet_r = foot_detect(positions, feet_thre)
    
    # 2. 計算旋轉與根節點速度
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    
    # 3. 計算相對位置 (RIC)
    # 將所有幀對齊到原點且初始朝向 Z+
    # 這裡的邏輯在 notebook 中是 get_rifke
    # 注意：這裡需要處理每幀的根節點位置與旋轉
    
    # 建立副本以進行座標變換
    local_positions = positions.copy()
    # 減去根節點的 XZ 位置
    local_positions[..., 0] -= positions[:, 0:1, 0]
    local_positions[..., 2] -= positions[:, 0:1, 2]
    # 繞 Y 軸旋轉對齊 (使每幀都朝向正前方)
    local_positions = qrot_np(np.repeat(r_rot[:, None], 22, axis=1), local_positions)
    
    # --- 開始組合 263D 向量 ---
    # A. 根節點特徵 (4D): [ang_vel(y), lin_vel(x), lin_vel(z), height]
    # r_velocity 是四元數，需要轉回弧度
    r_vel_y = np.arcsin(r_velocity[:, 2:3]) # 繞 Y 軸旋轉速度
    l_vel_xz = velocity[:, [0, 2]] # XZ 平面速度
    root_height = local_positions[:-1, 0, 1:2] # 根節點高度
    root_data = np.concatenate([r_vel_y, l_vel_xz, root_height], axis=-1) # (T-1, 4)
    
    # B. 相對位置 (63D): 21 joints * 3
    ric_data = local_positions[:-1, 1:].reshape(seq_len - 1, -1)
    
    # C. 旋轉特徵 (126D): 21 joints * 6
    rot_data = cont_6d_params[:-1, 1:].reshape(seq_len - 1, -1)
    
    # D. 線速度 (66D): 22 joints * 3
    # 這裡的速度是在局部座標系下
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], 22, axis=1), 
                        positions[1:] - positions[:-1])
    local_vel = local_vel.reshape(seq_len - 1, -1)
    
    # E. 腳底接觸 (4D)
    # feet_l, feet_r 已經是 (T-1, 2)
    
    # 組合所有特徵
    data = np.concatenate([root_data, ric_data, rot_data, local_vel, feet_l, feet_r], axis=-1)
    
    return data
