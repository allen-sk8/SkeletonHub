import numpy as np
import os
import sys
import argparse

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 匯入原本在 utils 中的恢復邏輯
from utils.humanml3d_utils import recover_from_ric

def convert_263d_to_joints(input_path, output_path=None):
    """
    將 HumanML3D 的 263D 特徵向量轉換回 3D 關節座標 (22 joints)
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到檔案: {input_path}")
        return None

    data = np.load(input_path)
    
    if data.shape[-1] != 263:
        print(f"❌ 維度不符：預期 263D，實際收到 {data.shape[-1]}D")
        return None

    print(f"🪄 正在轉換 {input_path} ...")
    joints = recover_from_ric(data)
    
    if output_path:
        np.save(output_path, joints)
        print(f"✅ 轉換完成，儲存至: {output_path}")
    
    return joints

if __name__ == "__main__":
    '''example
    python3 converters/humanml3d_263d_to_joints.py /home/allen/SkeletonHub/data/humanml3d/samples/012314.npy --output /home/allen/SkeletonHub/data/smpl_key_points/samples_22j/012314.npy
    '''
    parser = argparse.ArgumentParser(description="HumanML3D 263D to 22 Joints 轉換器")
    parser.add_argument("input", help="輸入的 .npy (263D) 路徑")
    parser.add_argument("--output", help="輸出的 .npy (Joints) 路徑")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        # 預設路徑到 /home/allen/SkeletonHub/data/smpl_key_points/samples_22j
        default_dir = "/home/allen/SkeletonHub/data/smpl_key_points/samples_22j"
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input)
        output = os.path.join(default_dir, base_name)
    
    convert_263d_to_joints(args.input, output)
