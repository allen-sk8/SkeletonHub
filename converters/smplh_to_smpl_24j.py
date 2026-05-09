"""
SMPL-H to Standard SMPL 24-Joint Converter

Usage:
    python converters/smplh_to_smpl_24j.py <input_smplh.pkl> [--output <output_24j.npy>] [--vis]

Technical Details & Data Sources:
    - Input: SMPL-H parameters (.pkl).
    - Skeletal mapping: Runs forward kinematics, maps to standard SMPL 24 joints by correctly selecting indices [:22] + [22] (left wrist) + [37] (right wrist) to resolve index offsets due to fingers.
    - Source: Standard SMPL Topology (https://smpl.is.tue.mpg.de/)
"""
import os
import numpy as np
import argparse
import pickle
import sys

# Example Usage:
# python converters/smplh_to_smpl_24j.py data/smpl/smplh/walking_01_poses.pkl --output data/smpl_joints/samples_24j/walking_01.npy

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.handler import SMPLHandler

def convert_smplh_to_smpl_24j(input_path, output_path=None):
    """
    將 SMPL-H 參數 (.pkl) 轉換為標準的 24 關節座標 (SMPL Standard)
    
    技術細節與索引映射邏輯：
    -------------------------------------------
    SMPL-H 模型 (52 joints) 的架構如下：
    - [0-21]: 身體主要關節 (與 Standard SMPL 一致)
    - [22-36]: 左手關節 (Left Hand tree)
    - [37-51]: 右手關節 (Right Hand tree)
    
    為了對齊 Standard SMPL (24 joints) 的定義：
    - [0-21]: 直接延用。
    - [22] (L_Hand): 對應 SMPL-H 的索引 22 (L_Thumb1)。
    - [23] (R_Hand): 對應 SMPL-H 的索引 37 (R_Thumb1)，需跳過左手其餘指節。
    
    參考來源：
    - MANO / SMPL-H 官方定義：https://mano.is.tue.mpg.de/
    - smplx 庫關節拓樸：smplx/body_models.py
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        return None

    # 1. 載入參數 (AMASS 格式 .pkl)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    poses = data['poses']
    betas = data['betas']
    trans = data['trans']
    gender = data['gender']

    # 2. 通過 SMPL 模型計算關節點 (執行 Forward Kinematics)
    # 調用 utils/smpl/handler.py，內部執行 AMASS Z-up 轉 Y-up 座標變換
    print(f"🚀 正在計算 SMPL-H (52j) 關節點...")
    handler = SMPLHandler(model_type='smplh')
    joints_52 = handler.params_to_joints(poses, betas, trans, gender)
    
    # 3. 執行精確索引映射 (Skeletal Decoupling)
    # 0-21 是身體主幹，22 是左手代表點 (SMPL-H 的 22)，23 是右手代表點 (SMPL-H 的 37)
    body_joints = joints_52[:, :22, :]
    l_hand = joints_52[:, 22:23, :]
    r_hand = joints_52[:, 37:38, :]
    joints_24 = np.concatenate([body_joints, l_hand, r_hand], axis=1)
    
    print(f"📊 擷取完成，座標維度: {joints_24.shape} (T, 24, 3)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints_24)
        print(f"✅ 轉換成功: {output_path}")

    return joints_24

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-H to SMPL 24j 轉換器")
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
    
    convert_smplh_to_smpl_24j(args.input, output)

    if args.vis:
        print("🎬 正在自動執行視覺化工具...")
        import subprocess
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        vis_script = os.path.join(project_root, "visualizers", "vis_smpl_joints.py")
        subprocess.run([sys.executable, vis_script, output])
