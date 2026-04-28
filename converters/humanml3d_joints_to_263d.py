import numpy as np
import os
import sys
import argparse
import torch

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.humanml3d_utils import extract_features
from utils.humanml3d_lib.skeleton import Skeleton
from utils.humanml3d_lib.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

def convert_joints_to_263d(input_path, output_path=None, feet_thre=0.002):
    """
    將 3D 關節座標 (22 joints) 轉換為 HumanML3D 的 263D 特徵向量
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        return None

    joints = np.load(input_path)
    
    # 檢查維度，預期 (T, 22, 3) 或 (T, 66)
    if joints.ndim == 2:
        joints = joints.reshape(len(joints), -1, 3)
    
    if joints.shape[1] != 22:
        print(f"❌ 關節數不符：預期 22 joints，實際收到 {joints.shape[1]}")
        return None

    print(f"🪄 正在從 {input_path} 提取特徵...")
    
    # 執行特徵提取
    features = extract_features(joints, feet_thre=feet_thre)
    
    print(f"📊 提取完成，特徵維度: {features.shape}")
    
    if output_path:
        # 確保目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        print(f"✅ 轉換完成，儲存至: {output_path}")
    
    return features

if __name__ == "__main__":
    '''example
    python3 converters/humanml3d_joints_to_263d.py data/smpl_key_points/samples_22j/012314.npy --output data/humanml3d/samples/012314_extracted.npy
    '''
    parser = argparse.ArgumentParser(description="22 Joints to HumanML3D 263D 轉換器")
    parser.add_argument("input", help="輸入的 .npy (Joints) 路徑")
    parser.add_argument("--output", help="輸出的 .npy (263D) 路徑")
    parser.add_argument("--feet_thre", type=float, default=0.002, help="腳底接觸偵測閾值 (預設: 0.002)")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        # 預設路徑到 data/humanml3d/samples/
        default_dir = "data/humanml3d/samples"
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input)
        output = os.path.join(default_dir, base_name)
    
    convert_joints_to_263d(args.input, output, feet_thre=args.feet_thre)
