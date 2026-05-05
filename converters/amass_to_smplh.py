import os
import numpy as np
import argparse
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.axis_converter import convert_smpl_z_to_y

def convert_amass_to_smplh(src_path, save_path, target_fps=20):
    """
    將 AMASS 的 .npz 檔案轉換為 SkeletonHub 標準的 SMPL-H (156j) .pkl 格式
    保留完整的手部參數
    """
    if not os.path.exists(src_path):
        print(f"❌ 找不到檔案: {src_path}")
        return

    try:
        bdata = np.load(src_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 讀取 .npz 失敗: {e}")
        return

    # 🌟 處理 FPS 缺失的情況
    fps = float(bdata.get('mocap_framerate', 20.0))
    down_sample = max(1, int(fps / target_fps))
    
    try:
        data = {
            'poses': bdata['poses'][::down_sample, :156].astype(np.float32),
            'trans': bdata['trans'][::down_sample, ...].astype(np.float32),
            'betas': bdata['betas'][:16].astype(np.float32), 
            'gender': str(bdata.get('gender', 'neutral')),
            'mocap_framerate': fps,
            'target_fps': target_fps,
            'source_path': os.path.abspath(src_path)
        }
        
        # 清理性別字串格式
        data['gender'] = data['gender'].replace("b'", "").replace("'", "").lower()
        if data['gender'] not in ['male', 'female', 'neutral']:
            data['gender'] = 'neutral'
        
        # 🌟 AMASS 是 Z-up，我們專案統一為 Y-up，呼叫統一轉換工具
        data = convert_smpl_z_to_y(data)
        
    except KeyError as e:
        print(f"❌ AMASS 檔案格式不符，缺少欄位: {e}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ 轉換成功 (SMPL-H 156D): {os.path.basename(src_path)}")
    print(f"   - 儲存路徑: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMASS .npz to SMPL-H (156D) .pkl 轉換器")
    parser.add_argument("input", help="輸入的 AMASS .npz 路徑")
    parser.add_argument("--output", help="輸出的 .pkl 路徑")
    parser.add_argument("--fps", type=int, default=20, help="目標影格率")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = os.path.join(os.getcwd(), "data", "smpl", "smplh")
        os.makedirs(default_dir, exist_ok=True)
        base_name = os.path.basename(args.input).replace(".npz", ".pkl")
        output = os.path.join(default_dir, base_name)
    
    convert_amass_to_smplh(args.input, output, target_fps=args.fps)
