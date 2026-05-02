import os
import sys
import argparse
import numpy as np

# Example Usage:
# python visualizers/vis_smpl_joints.py data/smpl_joints/samples_22j/walking_01_poses.npy
# python visualizers/vis_smpl_joints.py data/smpl_joints/samples_24j/walking_01_poses.npy
# python visualizers/vis_smpl_joints.py data/smpl_joints/samples_52j/walking_01_poses.npy

# 專案路徑匯入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.rendering.joints_renderer import render_motion, H3D_22_CHAIN, H3D_22_COLORS, SMPL_24_CHAIN, SMPL_24_COLORS

# --- 骨架定義庫 (額外擴充) ---

# 3. SMPL-H (52 Joints) - 含手指細節
SMPL_H_52_CHAIN = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], # Body
    [9, 14, 17, 19, 21], [9, 13, 16, 18, 20], # Arms
    # Right Hand Fingers
    [21, 37, 38, 39], [21, 40, 41, 42], [21, 43, 44, 45], [21, 46, 47, 48], [21, 49, 50, 51],
    # Left Hand Fingers
    [20, 22, 23, 24], [20, 25, 26, 27], [20, 28, 29, 30], [20, 31, 32, 33], [20, 34, 35, 36]
]
SMPL_H_52_COLORS = ['red', 'blue', 'black', 'red', 'blue'] + ['red']*5 + ['blue']*5

def main():
    parser = argparse.ArgumentParser(description="SMPL 家族關節座標視覺化程式 (支援 22j, 24j, 52j)")
    parser.add_argument("input", help="輸入的 .npy (T, J, 3) 檔案路徑")
    parser.add_argument("--fps", type=int, default=20, help="幀率")
    parser.add_argument("--radius", type=float, default=3.0, help="渲染半徑 (建議 3.0)")
    parser.add_argument("--name", help="自定義輸出檔名")
    
    args = parser.parse_args()

    # 1. 載入資料
    data = np.load(args.input)
    T, J, C = data.shape
    print(f"📊 偵測到數據維度: {data.shape} (Frames: {T}, Joints: {J})")

    # 2. 自動匹配骨架配置 (🌟 預設匹配將導向對應定義)
    if J == 22:
        print("🔗 匹配成功: HumanML3D (22 joints)")
        chain, colors = H3D_22_CHAIN, H3D_22_COLORS
    elif J == 24:
        print("🔗 匹配成功: Standard SMPL (24 joints)")
        chain, colors = SMPL_24_CHAIN, SMPL_24_COLORS
    elif J == 52:
        print("🔗 匹配成功: SMPL-H (52 joints)")
        chain, colors = SMPL_H_52_CHAIN, SMPL_H_52_COLORS
    else:
        print(f"⚠️ 未知的關節數 {J}，將使用 24j 預設連接方式 (可能出錯)")
        chain, colors = SMPL_24_CHAIN, SMPL_24_COLORS

    # 3. 設定輸出路徑
    input_dir = os.path.dirname(args.input)
    results_dir = os.path.join(input_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = args.name if args.name else os.path.basename(args.input).replace('.npy', '')
    output_path = os.path.join(results_dir, f"vis_{J}j_{base_name}.mp4")

    # 4. 執行渲染
    print(f"🚀 正在進行 {J}j 關節渲染至 {output_path} ...")
    render_motion(data, output_path, title=f"Joints ({J}j): {base_name}", 
                  fps=args.fps, radius=args.radius, kinematic_chain=chain, colors=colors)
    print("✅ 視覺化完成！")

if __name__ == "__main__":
    main()
