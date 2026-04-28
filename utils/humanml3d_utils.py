import torch
import numpy as np
import os
import sys

from .humanml3d_lib.skeleton import Skeleton
from .humanml3d_lib.quaternion import qrot, qinv, quaternion_to_cont6d, qbetween_np, qrot_np, qinv_np, qfix, qmul_np
from .humanml3d_lib.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

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
