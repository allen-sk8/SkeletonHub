import os
import argparse
import numpy as np
import pickle
import cv2
import torch
from scipy.spatial.transform import Rotation as R

def convert_joints_z_to_y(joints):
    """
    將 Z-up 關節座標轉換為 Y-up
    公式：new_pos = [x, z, -y] (繞 X 軸順時針旋轉 90 度，不產生鏡像)
    joints: (T, J, 3) 
    """
    converted = joints.copy()
    converted[..., 1] = joints[..., 2]
    converted[..., 2] = -joints[..., 1]
    return converted

def convert_smpl_z_to_y(data):
    """
    將 SMPL 參數從 Z-up 轉換為 Y-up
    只需旋轉 global_orient (Root Rotation) 與 transl (Root Translation)
    其餘 body_pose 是相對於父關節，不受全域旋轉影響。
    """
    converted = data.copy()
    
    # 1. 旋轉 Translation
    if 'trans' in converted:
        trans = converted['trans']
        new_trans = trans.copy()
        new_trans[..., 1] = trans[..., 2]
        new_trans[..., 2] = -trans[..., 1]
        converted['trans'] = new_trans
        
    # 2. 旋轉 Global Orient
    if 'poses' in converted:
        poses = converted['poses']
        global_orient = poses[:, :3]
        
        # 定義繞 X 軸旋轉 -90 度的旋轉矩陣 (Z-up to Y-up)
        # R_x(-90)
        r_x = R.from_euler('x', -90, degrees=True)
        
        # 原始 Root 旋轉
        r_orig = R.from_rotvec(global_orient)
        
        # 新的 Root 旋轉 = R_x * R_orig
        r_new = r_x * r_orig
        
        # 轉換回 axis-angle
        new_global_orient = r_new.as_rotvec()
        
        # 更新 poses
        converted['poses'][:, :3] = new_global_orient
        
    return converted

def main():
    parser = argparse.ArgumentParser(description="Z-up to Y-up 座標轉換工具")
    parser.add_argument("input", help="輸入檔案路徑 (.npy 或 .pkl)")
    parser.add_argument("--output", help="輸出檔案路徑 (若不指定則原地覆蓋)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output if args.output else input_path
    
    ext = os.path.splitext(input_path)[1].lower()
    
    print(f"🔄 準備轉換: {input_path}")
    print(f"📐 轉換邏輯: 繞 X 軸旋轉 -90 度 (新 Y = 舊 Z, 新 Z = -舊 Y)")
    
    if ext == '.npy':
        data = np.load(input_path)
        if len(data.shape) != 3 or data.shape[2] != 3:
            print("❌ 錯誤: .npy 檔案格式必須為 (T, J, 3)")
            return
        converted_data = convert_joints_z_to_y(data)
        np.save(output_path, converted_data)
        
    elif ext == '.pkl':
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        converted_data = convert_smpl_z_to_y(data)
        with open(output_path, 'wb') as f:
            pickle.dump(converted_data, f)
            
    else:
        print(f"❌ 錯誤: 不支援的檔案格式 {ext}")
        return
        
    print(f"✅ 轉換成功！結果儲存至: {output_path}")

if __name__ == "__main__":
    main()
