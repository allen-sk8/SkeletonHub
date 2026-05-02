import os
import sys
import argparse
import pickle
import numpy as np

# Example Usage:
# python visualizers/vis_smplh_mesh.py data/smpl/smplh/walking_01_poses.pkl --fps 20

# 專案路徑匯入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.smpl.handler import SMPLHandler
from utils.rendering.mesh_renderer import MeshRenderer

def main():
    parser = argparse.ArgumentParser(description="SMPL-H (156D) 參數 (.pkl) 視覺化入口程式")
    parser.add_argument("input", help="輸入的 .pkl 檔案路徑")
    parser.add_argument("--fps", type=int, default=20, help="幀率 (預設 20)")
    parser.add_argument("--name", help="自定義輸出檔名")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 找不到輸入檔案: {args.input}")
        return

    # 1. 載入數據
    print(f"🚀 [Step 1] 正在載入 SMPL-H 參數: {os.path.basename(args.input)}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    # 2. 計算頂點
    gender = data.get('gender', 'neutral')
    print(f"🚀 [Step 2] 正在通過 SMPL-H 模型計算 3D 頂點 (性別: {gender})...")
    
    try:
        handler = SMPLHandler(model_type="smplh")
        vertices, faces = handler.params_to_vertices(
            poses=data['poses'],
            betas=data['betas'],
            trans=data['trans'],
            gender=gender
        )
    except Exception as e:
        print(f"❌ SMPL-H 計算失敗: {e}")
        print("💡 提示: 請確保您下載的是「真正」包含手部組件的 SMPL-H 模型檔。")
        return
    
    # 3. 設定輸出路徑
    input_dir = os.path.dirname(args.input)
    results_dir = os.path.join(input_dir, 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = args.name if args.name else os.path.basename(args.input).replace('.pkl', '')
    output_path = os.path.join(results_dir, f"vis_smplh_{base_name}.mp4")

    # 4. 執行渲染
    print(f"🚀 [Step 3] 正在進行 Mesh 渲染至 {output_path} (Y-up 場景) ...")
    try:
        renderer = MeshRenderer()
        renderer.render_motion(vertices, faces, output_path, fps=args.fps, title=f"SMPL-H: {base_name}")
        print(f"✅ 視覺化完成！影片儲存於: {output_path}")
    except Exception as e:
        print(f"❌ 渲染失敗: {e}")

if __name__ == "__main__":
    main()
