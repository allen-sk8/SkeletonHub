import os
import pickle
import numpy as np
import argparse
import sys

# 將專案路徑加入以匯入工具 (如果需要)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.axis_converter import convert_joints_z_to_y

def convert_hybrik_to_joints_24j(input_path, output_path=None, scale=2.2):
    """
    從 HybrIK 輸出的 .pk 檔案中擷取 24 關節座標。
    優先使用 pred_xyz_24_struct_global，若無則使用 pred_xyz_24_struct。
    註：HybrIK 輸出的 3D 座標通常被歸一化至 [-1, 1] 的 2.2m Bounding Box，
    因此需預設乘上 2.2，將尺度還原為真實物理公尺 (meters)，否則 SMPL 渲染會因骨架太小而扭曲。
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        return None

    # 1. 載入 .pk 檔案 (通常是 pickle 格式)
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 讀取 {input_path} 失敗: {e}")
        # 有些 HybrIK 輸出可能是用 joblib 或 torch 存的，如果 pickle 失敗可以嘗試其他
        return None

    # 2. 擷取目標關節資料
    # 優先尋找 pred_xyz_24_struct_global
    if 'pred_xyz_24_struct_global' in data:
        joints = data['pred_xyz_24_struct_global']
        print(f"📊 擷取 pred_xyz_24_struct_global")
    elif 'pred_xyz_24_struct' in data:
        joints = data['pred_xyz_24_struct']
        print(f"📊 擷取 pred_xyz_24_struct (無 global 版本)")
    else:
        print(f"❌ 檔案中找不到 pred_xyz_24_struct_global 或 pred_xyz_24_struct")
        print(f"可用鍵值: {list(data.keys())}")
        return None

    # 確保是 numpy array
    joints = np.array(joints)
    
    # 3. 執行座標與尺度轉換 (Y-down to Y-up)
    joints[..., 1] *= -1  # 反轉 Y 軸 (修正上下顛倒)
    joints[..., 2] *= -1  # 反轉 Z 軸 (維持右手系)
    
    # 還原物理尺度
    joints *= scale
    print(f"📏 已應用尺度放大: x {scale}")

    # 檢查維度，預期是 (T, 24, 3)
    if joints.ndim == 3 and joints.shape[1] == 24 and joints.shape[2] == 3:
        print(f"✅ 成功擷取，維度: {joints.shape}")
    else:
        print(f"⚠️ 警告: 維度異常 {joints.shape}，預期為 (T, 24, 3)")

    # 3. 儲存結果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints.astype(np.float32))
        print(f"✅ 已存檔至: {output_path}")

    return joints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HybrIK .pk to 24j .npy 轉換器")
    parser.add_argument("input", help="輸入的 HybrIK .pk 檔案路徑")
    parser.add_argument("--output", help="輸出的 .npy 檔案路徑")
    parser.add_argument("--scale", type=float, default=2.2, help="座標放大倍率 (還原 HybrIK 2.2m Bounding Box 歸一化，預設: 2.2)")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    if not output_file:
        # 預設輸出路徑
        default_dir = "data/smpl_joints/samples_24j"
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(input_file).replace('.pk', '').replace('.pkl', '')
        output_file = os.path.join(default_dir, f"{base_name}_24j.npy")
    
    convert_hybrik_to_joints_24j(input_file, output_file, scale=args.scale)
