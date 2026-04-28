import os
import sys
import argparse
import numpy as np
'''example
python visualizers/vis_humanml3d.py /home/allen/SkeletonHub/data/humanml3d/samples/012314.npy
'''

# 專案路徑匯入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from converters.humanml3d_263d_to_joints import convert_263d_to_joints
from utils.renderer import render_motion

def main():
    parser = argparse.ArgumentParser(description="HumanML3D (263D) 視覺化入口程式")
    parser.add_argument("input", help="輸入的 .npy (263D) 檔案路徑")
    parser.add_argument("--fps", type=int, default=20, help="幀率 (預設 20)")
    parser.add_argument("--name", help="自定義輸出檔名")
    
    args = parser.parse_args()

    # 1. 執行格式轉換 (263D -> Joints)
    print("🚀 [Step 1] 正在將 HumanML3D 263D 特徵轉換為 22 節點關節座標...")
    joints = convert_263d_to_joints(args.input)
    
    if joints is None:
        print("❌ 轉換失敗，終止視覺化。")
        return

    # 2. 設定輸出路徑：輸入路徑的同個資料夾下再開一個 visualizations 資料夾
    input_dir = os.path.dirname(args.input)
    results_dir = os.path.join(input_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = args.name if args.name else os.path.basename(args.input).replace('.npy', '')
    output_path = os.path.join(results_dir, f"vis_humanml3d_{base_name}.mp4")

    # 3. 執行渲染
    print(f"🚀 [Step 2] 正在渲染 3D 動作至 {output_path} ...")
    render_motion(joints, output_path, title=f"HumanML3D: {base_name}", fps=args.fps)
    print("✅ 視覺化完成！")

if __name__ == "__main__":
    main()
