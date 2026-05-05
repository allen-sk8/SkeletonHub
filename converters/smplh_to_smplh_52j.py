import os
import numpy as np
import argparse
import pickle
import sys

# Example Usage:
# python converters/smplh_to_smplh_52j.py data/smpl/smplh/walking_01_poses.pkl --output data/smpl_joints/samples_52j/walking_01.npy

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.handler import SMPLHandler

def convert_smplh_to_smplh_52j(input_path, output_path=None):
    """
    將 SMPL-H 參數 (.pkl) 轉換為完整的 52 關節座標 (含手部)
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
    joints_full = handler.params_to_joints(poses, betas, trans, gender)
    
    # 3. 擷取前 52 個關節
    joints_52 = joints_full[:, :52, :]
    
    print(f"📊 提取完成，座標維度: {joints_52.shape} (T, 52, 3)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints_52)
        print(f"✅ 轉換成功: {output_path}")

    return joints_52

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-H to 52j (Full Skeleton) 轉換器")
    parser.add_argument("input", help="輸入的 .pkl 檔案路徑")
    parser.add_argument("--output", help="輸出的 .npy 檔案路徑")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = "data/smpl_joints/samples_52j"
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input).replace('.pkl', '')
        output = os.path.join(default_dir, f"52j_{base_name.replace('smplh_', '')}.npy")
    
    convert_smplh_to_smplh_52j(args.input, output)
