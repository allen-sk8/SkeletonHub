"""
SMPL to SMPL 24-Joint Converter

Usage:
    python converters/smpl_to_smpl_24j.py <input_smpl.pkl> [--output <output_24j.npy>] [--vis]

Technical Details & Data Sources:
    - Input: SMPL parameters (.pkl).
    - Skeletal mapping: Runs forward kinematics to output full 24 joints (body only) of standard SMPL skeleton hierarchy.
    - Source: SMPL Model (https://mano.is.tue.mpg.de/)
"""
import os
import numpy as np
import argparse
import pickle
import sys

# Example Usage:
# python converters/smpl_to_smpl_24j.py data/smpl/smpl/walking_01_poses.pkl --output data/smpl_joints/samples_24j/walking_01.npy

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.handler import SMPLHandler

def convert_smpl_to_smpl_24j(input_path, output_path=None):
    """
    將 SMPL 參數 (.pkl) 轉換為完整的 24 關節座標 (body only)
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
    print(f"🚀 正在計算 SMPL (24j) 關節點...")
    handler = SMPLHandler(model_type='smpl')
    joints_full = handler.params_to_joints(poses, betas, trans, gender)
    
    # 3. 擷取前 24 個關節
    joints_24 = joints_full[:, :24, :]
    
    print(f"📊 提取完成，座標維度: {joints_24.shape} (T, 24, 3)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints_24)
        print(f"✅ 轉換成功: {output_path}")

    return joints_24

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL to 24j (Body Only) 轉換器")
    parser.add_argument("input", help="輸入的 .pkl 檔案路徑")
    parser.add_argument("--output", help="輸出的 .npy 檔案路徑")
    parser.add_argument("--vis", action="store_true", help="是否自動跑視覺化工具")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = "data/smpl_joints/samples_24j"
        os.makedirs(default_dir, exist_ok=True)
        base_name, _ = os.path.splitext(os.path.basename(args.input))
        output = os.path.join(default_dir, f"{base_name}_24j.npy")
    
    convert_smpl_to_smpl_24j(args.input, output)

    if args.vis:
        print("🎬 正在自動執行視覺化工具...")
        import subprocess
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        vis_script = os.path.join(project_root, "visualizers", "vis_smpl_joints.py")
        subprocess.run([sys.executable, vis_script, output])
