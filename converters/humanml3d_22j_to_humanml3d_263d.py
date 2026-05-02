import numpy as np
import os
import sys
import argparse
import torch

# Example Usage:
# python converters/humanml3d_22j_to_humanml3d_263d.py data/smpl_joints/samples_22j/012314.npy --output data/humanml3d/samples/012314.npy

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.humanml3d.utils import extract_features

def convert_humanml3d_22j_to_humanml3d_263d(input_path, output_path=None, feet_thre=0.002):
    """
    將 3D 關節座標 (22 joints) 轉換為 HumanML3D 的 263D 特徵向量
    
    技術規格與 263D 維度拆解：
    -------------------------------------------
    總維度：263
    1. Root Velocity (4D): [r_vel_y, l_vel_x, l_vel_z, root_height]
    2. RIC (Local Positions, 63D): 21 joints * 3 (Relative to root)
    3. Rotation (Continuous 6D, 126D): 21 joints * 6
    4. Linear Velocity (Local, 66D): 22 joints * 3
    5. Foot Contacts (4D): [L_Heel, L_Toe, R_Heel, R_Toe]
    
    參考實作：
    - 來源倉庫：https://github.com/GuoAndong/HumanML3D
    - 關鍵函數：`extract_features` (位於 utils/humanml3d/utils.py)
    - 邏輯源頭：`external/HumanML3D/motion_representation.ipynb` 中的特徵提取流水線。
    
    座標處理細節：
    - 執行 `get_rifke` 旋轉對齊：所有幀均繞 Y 軸旋轉以對準第一幀的朝向。
    - 物理單位：公尺 (m)。
    - FPS：對應原始數據 FPS (通常為 20Hz)。
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        return None

    # 1. 載入 22j 數據 (T, 22, 3)
    joints = np.load(input_path)
    print(f"🪄 正在從 {input_path} 提取 263D 特徵...")
    
    # 2. 調用核心提取邏輯 (移植自 HumanML3D 官方代碼)
    features = extract_features(joints, feet_thre=feet_thre)
    
    print(f"📊 提取完成，特徵維度: {features.shape}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        print(f"✅ 轉換完成，儲存至: {output_path}")
    
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanML3D 22j to 263D 轉換器")
    parser.add_argument("input", help="輸入路徑 (.npy, 22j)")
    parser.add_argument("--output", help="輸出路徑 (.npy, 263D)")
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = "data/humanml3d/samples"
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input)
        output = os.path.join(default_dir, base_name)
        
    convert_humanml3d_22j_to_humanml3d_263d(args.input, output)
