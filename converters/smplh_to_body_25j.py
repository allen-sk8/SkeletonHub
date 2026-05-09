"""
SMPL-H to OpenPose BODY25 Converter

Usage:
    python converters/smplh_to_body_25j.py <input_smplh.pkl> [--output <output_body25.npy>] [--regressor <regressor_path>] [--vis]

Technical Details & Data Sources:
    - Input: SMPL-H parameters (.pkl).
    - Vertices regression: Converts SMPL-H parameters to 6890 mesh vertices via SMPLHandler, then maps them to OpenPose BODY25 joints via J_regressor_body25.npy.
    - Source: Joint Regressor from EasyMocap (https://github.com/zju3dv/EasyMocap)
"""
import os
import numpy as np
import argparse
import pickle
import sys

# Example Usage:
# python converters/smplh_to_body_25j.py data/smpl/smplh/walking_01_poses.pkl --output data/body25/walking_01.npy

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.handler import SMPLHandler

def convert_smplh_to_body_25j(input_path, output_path=None, regressor_path=None):
    """
    將 SMPL-H 參數 (.pkl) 轉換為 OpenPose BODY25 格式的 25 關節 3D 座標。
    
    技術背景與數據來源記錄：
    -------------------------------------------
    - 轉回 BODY25 格式不採用單純的關節拓樸映射，而是使用關節回歸器 (Joint Regressor)。
    - 本工具加載 J_regressor_body25.npy 來將 SMPL 的 6890 個頂點 (Vertices) 線性回歸到 BODY25 的 25 個關節。
    - 外部數據/工具來源：
      * 來源：EasyMocap (https://github.com/zju3dv/EasyMocap)
      * 原始下載網址：https://github.com/zju3dv/EasyMocap/raw/master/data/smplx/J_regressor_body25.npy
      * 本地儲存路徑：common_models/regressor/J_regressor_body25.npy
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到輸入檔案: {input_path}")
        return None

    # 1. 確定回歸器路徑
    if regressor_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        regressor_path = os.path.join(project_root, 'common_models', 'regressor', 'J_regressor_body25.npy')

    if not os.path.exists(regressor_path):
        print(f"❌ 找不到 Body25 回歸矩陣: {regressor_path}")
        # 自動嘗試備用途徑
        fallback_path = os.path.join(project_root, 'external', 'EasyMocap', 'data', 'smplx', 'J_regressor_body25.npy')
        if os.path.exists(fallback_path):
            regressor_path = fallback_path
            print(f"ℹ️ 使用備用 EasyMocap 回歸矩陣路徑: {regressor_path}")
        else:
            print("❌ 請確保 common_models/regressor/J_regressor_body25.npy 存在")
            return None

    # 2. 載入 SMPL-H 參數
    print(f"📂 載入參數檔案: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    poses = data['poses']
    betas = data['betas']
    trans = data['trans']
    gender = data['gender']

    # 3. 載入回歸矩陣
    print(f"⚙️ 載入回歸矩陣: {regressor_path}")
    regressor = np.load(regressor_path) # Shape: (25, 6890)

    # 4. 透過 SMPL 模型計算 6890 個 3D 頂點
    print(f"🚀 正在計算 SMPL-H 3D 頂點 (Vertices)...")
    handler = SMPLHandler(model_type='smplh')
    vertices, faces = handler.params_to_vertices(poses, betas, trans, gender) # Shape: (T, 6890, 3)
    
    # 5. 線性回歸計算 body25 3D 關節
    print(f"📐 正在透過回歸矩陣計算 BODY25 關節...")
    joints_25 = np.matmul(regressor[None], vertices) # Shape: (T, 25, 3)
    print(f"📊 轉換完成，座標維度: {joints_25.shape} (T, 25, 3)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints_25)
        print(f"✅ 轉換成功，已儲存至: {output_path}")

    return joints_25

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-H to OpenPose BODY25 轉換器")
    parser.add_argument("input", help="輸入的 .pkl 檔案路徑")
    parser.add_argument("--output", help="輸出的 .npy 檔案路徑")
    parser.add_argument("--regressor", help="自定義 J_regressor_body25.npy 回歸矩陣路徑")
    parser.add_argument("--vis", action="store_true", help="是否自動跑視覺化工具")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = "data/body25"
        os.makedirs(default_dir, exist_ok=True)
        base_name, _ = os.path.splitext(os.path.basename(args.input))
        output = os.path.join(default_dir, f"{base_name}_body25.npy")
    
    convert_smplh_to_body_25j(args.input, output, args.regressor)

    if args.vis:
        print("🎬 正在自動執行視覺化工具...")
        import subprocess
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        vis_script = os.path.join(project_root, "visualizers", "vis_body25_joints.py")
        subprocess.run([sys.executable, vis_script, output])
