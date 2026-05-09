import os
import sys
import torch
import numpy as np
import pickle
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.easymocap_wrapper import EasyMocapWrapper

def main():
    parser = argparse.ArgumentParser(description="Fit SMPL-H (52j) to 3D joints using EasyMocap.")
    parser.add_argument("input", help="Path to input .npy joints (T, 52, 3)")
    parser.add_argument("--output", help="Path to output .pkl")
    parser.add_argument("--no_hand", action='store_true', help="Disable hand fitting (not recommended for 52j)")
    parser.add_argument("--vis", action="store_true", help="是否自動跑視覺化工具")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    # 1. Load data
    target_joints = np.load(args.input)
    print(f"📦 Loaded 52j joints with shape: {target_joints.shape}")
    
    if target_joints.shape[1] < 52:
        print(f"⚠️ Warning: Expected smpl-h 52 joints, but got {target_joints.shape[1]}. Fitting might be inaccurate.")

    # 2. Initialize Wrapper (Forced to smplh for 52j)
    wrapper = EasyMocapWrapper(model_type='smplh')
    
    # 3. Perform Fitting
    use_hand = not args.no_hand
    result = wrapper.fit_3d(target_joints, use_hand=use_hand)
    
    # 4. Save result
    out_path = args.output
    if not out_path:
        os.makedirs("data/smpl/smplh", exist_ok=True)
        base_name, _ = os.path.splitext(os.path.basename(args.input))
        out_path = os.path.join("data/smpl/smplh", f"{base_name}_smplh.pkl")
        
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
        
    print(f"✅ 52j Fitting complete! Result saved to: {out_path}")

    if args.vis:
        print("🎬 正在自動執行視覺化工具...")
        import subprocess
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        vis_script = os.path.join(project_root, "visualizers", "vis_smplh_mesh.py")
        subprocess.run([sys.executable, vis_script, out_path])

if __name__ == "__main__":
    main()
