import numpy as np
import os
import sys
import argparse

# Example Usage:
# python converters/humanml3d_263d_to_humanml3d_22j.py data/humanml3d/samples/012314.npy --output data/smpl_joints/samples_22j/012314.npy

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.humanml3d.utils import recover_from_ric

def convert_humanml3d_263d_to_humanml3d_22j(input_path, output_path=None):
    """
    將 HumanML3D 的 263D 特徵向量轉換回 3D 關節座標 (22 joints)
    
    逆向恢復邏輯拆解 (Inverse Pipeline)：
    -------------------------------------------
    1. 根節點恢復 (`recover_root_rot_pos`)：
       - 從特徵 index 0 提取旋轉速度，並透過 `np.cumsum` 積分得到各幀偏航角 (Yaw)。
       - 從特徵 index 1, 2 提取水平速度，並透過旋轉矩陣變換回全局座標系後積分得到 XZ 位移。
       - 從特徵 index 3 直接取得根節點高度 (Y)。
    2. 局部座標恢復 (RIC)：
       - 提取 index 4 到 66 (21*3) 的局部座標。
       - 將局部座標繞 Y 軸旋轉回世界座標系，並疊加已恢復的根節點位置。
    
    參考實作：
    - 來源腳本：`utils/humanml3d/utils.py` 中的 `recover_from_ric` 與 `recover_root_rot_pos`。
    - 理論出處：HumanML3D 論文中的 Motion Representation 章節。
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        return None

    # 1. 載入 263D 特徵數據
    data = np.load(input_path)
    print(f"🪄 正在從 {input_path} 恢復 22 關節座標...")
    
    # 2. 執行逆向恢復邏輯
    joints = recover_from_ric(data, joints_num=22)
    
    print(f"📊 恢復完成，座標維度: {joints.shape} (T, 22, 3)")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints)
        print(f"✅ 轉換完成，儲存至: {output_path}")
    
    return joints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanML3D 263D to 22j 轉換器")
    parser.add_argument("input", help="輸入路徑 (.npy, 263D)")
    parser.add_argument("--output", help="輸出路徑 (.npy, 22j)")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = "data/smpl_joints/samples_22j"
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input)
        output = os.path.join(default_dir, "22j_" + base_name.replace("263d_", ""))
        
    convert_humanml3d_263d_to_humanml3d_22j(args.input, output)
