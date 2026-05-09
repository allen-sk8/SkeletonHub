"""
24-Joint 3D coordinates to Standard SMPL Parameter Converter (EasyMocap Optimization)

Usage:
    python converters/joints_24j_to_smpl.py <input_24j.npy> [--output <output_smpl.pkl>] [--model smpl] [--vis]

Technical Details & Data Sources:
    - Input: 3D Joint coordinates (24 joints, Shape: T, 24, 3) in Y-up.
    - Optimization fitting: Formulates and solves SMPL/SMPL-H parameter regression using EasyMocap fitting engine.
    - Source: EasyMocap (https://github.com/zju3dv/EasyMocap)
"""
import os
import sys
import numpy as np
import pickle
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.easymocap_wrapper import EasyMocapWrapper

def main():
    parser = argparse.ArgumentParser(description="24j XYZ → SMPL 參數 (EasyMocap fitting)")
    parser.add_argument("input", help="輸入的 .npy joints 檔案 (T, 24, 3)")
    parser.add_argument("--output", help="輸出的 .pkl 檔案路徑")
    parser.add_argument("--model", default="smpl", choices=["smpl", "smplh"],
                        help="SMPL 模型類型 (預設: smpl)")
    parser.add_argument("--vis", action="store_true", help="是否自動跑視覺化工具")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    # 1. Load data
    target_joints = np.load(args.input)
    print(f"📦 Loaded joints with shape: {target_joints.shape}")
    
    if target_joints.ndim != 3 or target_joints.shape[1] != 24 or target_joints.shape[2] != 3:
        print(f"❌ 預期維度為 (T, 24, 3)，實際為 {target_joints.shape}")
        return
    
    # 2. Initialize EasyMocap wrapper
    wrapper = EasyMocapWrapper(model_type=args.model)
    
    # 3. Perform fitting
    result = wrapper.fit_3d(target_joints)
    
    # 4. Save result
    out_path = args.output
    if not out_path:
        os.makedirs("data/smpl/smpl", exist_ok=True)
        base_name, _ = os.path.splitext(os.path.basename(args.input))
        out_path = os.path.join("data/smpl/smpl", f"{base_name}_smpl.pkl")
        
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
        
    print(f"✅ Fitting complete! Result saved to: {out_path}")

    if args.vis:
        print("🎬 正在自動執行視覺化工具...")
        import subprocess
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        vis_script = os.path.join(project_root, "visualizers", "vis_smpl_mesh.py")
        subprocess.run([sys.executable, vis_script, out_path])

if __name__ == "__main__":
    main()
