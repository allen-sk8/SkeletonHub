"""
COCO-Wholebody (133 joints) to OpenPose BODY25 Converter

Usage:
    python converters/joints_133j_to_body_25j.py <input_joints133.npy> [--output <output_body25.npy>] [--vis]
"""
import os
import sys
import argparse
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def convert_joints_133j_to_body_25j(input_path, output_path=None):
    """
    將 COCO-Wholebody 133 關節座標 (.npy) 轉換為 OpenPose BODY25 格式的 25 關節 3D 座標。
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到輸入檔案: {input_path}")
        return None

    print(f"📂 載入關節座標檔: {input_path}")
    data = np.load(input_path)
    
    if data.ndim != 3 or data.shape[2] != 3:
        print(f"❌ 數據維度錯誤: 預期為 (T, 133, 3)，但得到 {data.shape}")
        return None
        
    T, J, C = data.shape
    if J != 133:
        print(f"⚠️ 警告: 輸入的關節數為 {J}，非標準 COCO-Wholebody (133 關節)。將強制對應前 23 個關節點。")

    # 建立 BODY25 關節陣列
    body25_joints = np.zeros((T, 25, 3), dtype=np.float32)

    # 1. 直接索引映射 (Direct Index Mapping)
    # COCO-Wholebody 與 BODY25 共通的直接對應點
    body25_joints[:, 0] = data[:, 0]    # Nose -> Nose
    body25_joints[:, 2] = data[:, 6]    # RShoulder -> RShoulder
    body25_joints[:, 3] = data[:, 8]    # RElbow -> RElbow
    body25_joints[:, 4] = data[:, 10]   # RWrist -> RWrist
    body25_joints[:, 5] = data[:, 5]    # LShoulder -> LShoulder
    body25_joints[:, 6] = data[:, 7]    # LElbow -> LElbow
    body25_joints[:, 7] = data[:, 9]    # LWrist -> LWrist
    body25_joints[:, 9] = data[:, 12]   # RHip -> RHip
    body25_joints[:, 10] = data[:, 14]  # RKnee -> RKnee
    body25_joints[:, 11] = data[:, 16]  # RAnkle -> RAnkle
    body25_joints[:, 12] = data[:, 11]  # LHip -> LHip
    body25_joints[:, 13] = data[:, 13]  # LKnee -> LKnee
    body25_joints[:, 14] = data[:, 15]  # LAnkle -> LAnkle
    body25_joints[:, 15] = data[:, 2]    # REye -> REye
    body25_joints[:, 16] = data[:, 1]    # LEye -> LEye
    body25_joints[:, 17] = data[:, 4]    # REar -> REar
    body25_joints[:, 18] = data[:, 3]    # LEar -> LEar
    
    # 雙腳指與腳跟映射 (得益於 COCO-Wholebody 對 Foot 的標註，我們有精準的對應)
    body25_joints[:, 19] = data[:, 17]   # LBigToe -> LBigToe
    body25_joints[:, 20] = data[:, 18]   # LSmallToe -> LSmallToe
    body25_joints[:, 21] = data[:, 19]   # LHeel -> LHeel
    body25_joints[:, 22] = data[:, 20]   # RBigToe -> RBigToe
    body25_joints[:, 23] = data[:, 21]   # RSmallToe -> RSmallToe
    body25_joints[:, 24] = data[:, 22]   # RHeel -> RHeel

    # 2. 插值映射 (Interpolated Joints)
    # Neck (BODY25 index 1) = (LShoulder[5] + RShoulder[6]) / 2
    body25_joints[:, 1] = (data[:, 5] + data[:, 6]) / 2.0
    
    # MidHip (BODY25 index 8) = (LHip[11] + RHip[12]) / 2
    body25_joints[:, 8] = (data[:, 11] + data[:, 12]) / 2.0

    print(f"📐 轉換完成，輸出座標維度: {body25_joints.shape} (T, 25, 3)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, body25_joints)
        print(f"✅ 成功儲存 BODY25 格式檔案至: {output_path}")

    return body25_joints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO-Wholebody 133 關節 ➔ OpenPose BODY25 轉換器")
    parser.add_argument("input", help="輸入的 .npy (133 關節) 檔案路徑")
    parser.add_argument("--output", help="輸出的 .npy (25 關節) 檔案路徑")
    parser.add_argument("--vis", action="store_true", help="是否自動跑 BODY25 視覺化工具")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        # 自動後綴追加與路徑切換
        default_dir = os.path.join(project_root, "data", "body25")
        os.makedirs(default_dir, exist_ok=True)
        
        # 移除原有的 _coco_wholebody133 或 _joints133，改為 _body25
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        if base_name.endswith("_coco_wholebody133"):
            base_name = base_name[:-18]
        elif base_name.endswith("_joints133"):
            base_name = base_name[:-10]
        output = os.path.join(default_dir, f"{base_name}_body25.npy")
        
    convert_joints_133j_to_body_25j(args.input, output)
    
    if args.vis:
        print("🎬 正在自動執行 BODY25 視覺化工具...")
        import subprocess
        vis_script = os.path.join(project_root, "visualizers", "vis_body25_joints.py")
        subprocess.run([sys.executable, vis_script, output])
