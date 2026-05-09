import os
import sys
import argparse
import numpy as np

# Example Usage:
# python visualizers/vis_body25_joints.py data/body25/walking_01.npy

# 專案路徑匯入以導入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.rendering.joints_renderer import render_motion

# --- OpenPose BODY25 骨架拓撲與鏈條定義 ---
# 定義 3D 關節渲染連接順序，讓人體骨架連線看起來自然且符合 OpenPose BODY25 索引規格：
# 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist, 
# 8: MidHip, 9: RHip, 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle, 15: REye, 
# 16: LEye, 17: REar, 18: LEar, 19: LBigToe, 20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel
BODY_25_CHAIN = [
    [8, 1, 0, 15, 17],        # 脊椎 ➔ 脖子 ➔ 鼻子 ➔ 右眼 ➔ 右耳 (中軸與右側頭部)
    [0, 16, 18],              # 鼻子 ➔ 左眼 ➔ 左耳 (左側頭部)
    [1, 2, 3, 4],             # 脖子 ➔ 右肩 ➔ 右肘 ➔ 右手腕 (右手臂)
    [1, 5, 6, 7],             # 脖子 ➔ 左肩 ➔ 左肘 ➔ 左手腕 (左手臂)
    [8, 9, 10, 11, 22, 23],   # 臀部 ➔ 右髖 ➔ 右膝 ➔ 右踝 ➔ 右大腳趾 ➔ 右小腳趾 (右腿+前掌)
    [11, 24],                 # 右踝 ➔ 右腳跟
    [8, 12, 13, 14, 19, 20],  # 臀部 ➔ 左髖 ➔ 左膝 ➔ 左踝 ➔ 左大腳趾 ➔ 左小腳趾 (左腿+前掌)
    [14, 21]                  # 左踝 ➔ 左腳跟
]

# 左右與中軸連線顏色 (符合 SkeletonHub 的規範: Right 用紅色, Left 用藍色, Center 用黑色)
BODY_25_COLORS = [
    'black',  # 中軸軀幹 & 右眼耳
    'black',  # 左眼耳
    'red',    # 右手臂
    'blue',   # 左手臂
    'red',    # 右腿 & 右腳趾
    'red',    # 右腳跟
    'blue',   # 左腿 & 左腳趾
    'blue'    # 左腳跟
]

def main():
    parser = argparse.ArgumentParser(description="OpenPose BODY25 格式 (25 關節) 座標視覺化程式")
    parser.add_argument("input", help="輸入的 .npy (T, 25, 3) 檔案路徑")
    parser.add_argument("--fps", type=int, default=20, help="幀率 (預設: 20)")
    parser.add_argument("--radius", type=float, default=3.0, help="渲染半徑 (預設: 3.0)")
    parser.add_argument("--name", help="自定義輸出影片檔名")
    
    args = parser.parse_args()

    # 1. 載入關節點資料
    if not os.path.exists(args.input):
        print(f"❌ 找不到輸入檔案: {args.input}")
        sys.exit(1)
        
    data = np.load(args.input)
    if data.ndim != 3:
        print(f"❌ 數據維度錯誤: 預期為 (T, J, 3)，但得到 {data.shape}")
        sys.exit(1)
        
    T, J, C = data.shape
    print(f"📊 偵測到數據維度: {data.shape} (Frames: {T}, Joints: {J})")

    # 2. 驗證是否為 25 關節
    if J != 25:
        print(f"⚠️ 警告: 偵測到關節數為 {J}，非標準的 BODY25 (25 關節)。將強制套用 BODY25 拓撲連線進行渲染，可能會產生異常。")

    # 3. 設定輸出路徑
    input_dir = os.path.dirname(args.input)
    results_dir = os.path.join(input_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = args.name if args.name else os.path.basename(args.input).replace('.npy', '')
    output_path = os.path.join(results_dir, f"vis_body25_{base_name}.mp4")

    # 4. 執行動作渲染
    print(f"🚀 正在進行 BODY25 關節渲染至 {output_path} ...")
    render_motion(
        data, 
        output_path, 
        title=f"OpenPose BODY25: {base_name}", 
        fps=args.fps, 
        radius=args.radius, 
        kinematic_chain=BODY_25_CHAIN, 
        colors=BODY_25_COLORS
    )
    print("✅ BODY25 視覺化完成！")

if __name__ == "__main__":
    main()
