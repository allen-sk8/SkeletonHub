import os
import sys
import argparse
import numpy as np
'''example
python3 visualizers/vis_joints.py "/home/allen/SkeletonHub/data/smpl_key_points/samples/012314.npy"
'''
# 專案路徑匯入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.renderer import render_motion

def main():
    parser = argparse.ArgumentParser(description="通用 3D 關節座標視覺化入口程式")
    parser.add_argument("input", help="輸入的 .npy (T, J, 3) 檔案路徑")
    parser.add_argument("--fps", type=int, default=20, help="幀率 (預設 20)")
    parser.add_argument("--name", help="自定義輸出檔名")
    
    args = parser.parse_args()

    # 1. 載入資料並檢查維度
    data = np.load(args.input)
    if len(data.shape) != 3 or data.shape[-1] != 3:
        print(f"❌ 錯誤：不支援的維度 {data.shape}。此程式僅支援 (T, J, 3) 格式。")
        return

    # 2. 設定輸出路徑：輸入路徑的同個資料夾下再開一個 visualizations 資料夾
    input_dir = os.path.dirname(args.input)
    results_dir = os.path.join(input_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = args.name if args.name else os.path.basename(args.input).replace('.npy', '')
    output_path = os.path.join(results_dir, f"vis_joints_{base_name}.mp4")

    # 3. 執行渲染
    print(f"🚀 正在渲染 3D 關節動作至 {output_path} ...")
    render_motion(data, output_path, title=f"Joints: {base_name}", fps=args.fps)
    print("✅ 視覺化完成！")

if __name__ == "__main__":
    main()
