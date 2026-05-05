import os
import numpy as np
import argparse
import pickle
import sys

# Example Usage:
# python converters/smplh_to_humanml3d_22j.py data/smpl/smplh/walking_01_poses.pkl --output data/smpl_joints/samples_22j/walking_01.npy

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.handler import SMPLHandler

def convert_smplh_to_humanml3d_22j(input_path, output_path=None):
    """
    將 SMPL-H 參數 (.pkl) 轉換為 22 關節座標 (HumanML3D Style)
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        return None

    # 1. 載入參數
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    poses = data['poses']
    betas = data['betas']
    trans = data['trans']
    gender = data['gender']

    # 2. 通過 SMPL 模型計算關節點
    print(f"🚀 正在計算 SMPL-H (52j) 關節點...")
    handler = SMPLHandler(model_type='smplh')
    joints_52 = handler.params_to_joints(poses, betas, trans, gender)
    
    # 3. 擷取前 22 個關節 (HumanML3D 使用的子集)
    joints_22 = joints_52[:, :22, :]
    
    print(f"📊 擷取完成，座標維度: {joints_22.shape} (T, 22, 3)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints_22)
        print(f"✅ 轉換成功: {output_path}")

    return joints_22

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-H to HumanML3D 22j 轉換器")
    parser.add_argument("input", help="輸入的 .pkl 檔案路徑")
    parser.add_argument("--output", help="輸出的 .npy 檔案路徑")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = "data/smpl_joints/samples_22j"
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input).replace('.pkl', '')
        output = os.path.join(default_dir, f"22j_{base_name.replace('smplh_', '')}.npy")
    
    convert_smplh_to_humanml3d_22j(args.input, output)
