"""
AMASS to Standard SMPL 72D .pkl Converter

Usage:
    python converters/amass_to_smpl.py <input_amass.npz> [--output <output_smpl.pkl>] [--fps 20] [--vis]

Technical Details & Data Sources:
    - Input: AMASS Dataset .npz file (Z-up, SMPL/SMPL-H compatible).
    - Dimension extraction: Slices first 72 values of pose (global_orient + 23 joints) and first 10 values of betas.
    - Coordinate conversion: Calls convert_smpl_z_to_y to project from Z-up coordinate system to standard Y-up.
    - Source: AMASS Dataset (https://amass.is.tue.mpg.de/)
"""
import os
import numpy as np
import argparse
import pickle
import sys

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.axis_converter import convert_smpl_z_to_y

def convert_amass_to_smpl(src_path, save_path, target_fps=20):
    """
    將 AMASS 的 .npz 檔案轉換為 SkeletonHub 標準的 SMPL (72j) .pkl 格式
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
        # standard SMPL uses first 72 parameters for poses and first 10 for betas
        data = {
            'poses': bdata['poses'][::down_sample, :72].astype(np.float32),
            'trans': bdata['trans'][::down_sample, ...].astype(np.float32),
            'betas': bdata['betas'][:10].astype(np.float32), 
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
    
    print(f"✅ 轉換成功 (SMPL 72D): {os.path.basename(src_path)}")
    print(f"   - 儲存路徑: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMASS .npz to SMPL (72D) .pkl 轉換器")
    parser.add_argument("input", help="輸入的 AMASS .npz 路徑")
    parser.add_argument("--output", help="輸出的 .pkl 路徑")
    parser.add_argument("--fps", type=int, default=20, help="目標影格率")
    parser.add_argument("--vis", action="store_true", help="是否自動跑視覺化工具")
    
    args = parser.parse_args()
    
    output = args.output
    if not output:
        default_dir = os.path.join(os.getcwd(), "data", "smpl", "smpl")
        os.makedirs(default_dir, exist_ok=True)
        base_name, _ = os.path.splitext(os.path.basename(args.input))
        output = os.path.join(default_dir, f"{base_name}_smpl.pkl")
    
    convert_amass_to_smpl(args.input, output, target_fps=args.fps)
    
    if args.vis:
        print("🎬 正在自動執行視覺化工具...")
        import subprocess
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        vis_script = os.path.join(project_root, "visualizers", "vis_smpl_mesh.py")
        subprocess.run([sys.executable, vis_script, output])
