"""
24 Joint XYZ → SMPL 參數轉換器

使用自建的 SMPL IK Solver 進行逆向運動學求解。
輸入：(T, 24, 3) 的 Y-up 公尺制 3D 關節座標
輸出：標準 SMPL .pkl (poses, betas, trans, gender)

用法:
    python converters/joints_24j_to_smpl.py data/smpl_joints/samples_24j/some_motion.npy
    python converters/joints_24j_to_smpl.py input.npy --output output.pkl --gender male
"""
import os
import sys
import numpy as np
import pickle
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.smpl_ik_solver import SMPLIKSolver

def main():
    parser = argparse.ArgumentParser(description="24j XYZ → SMPL 參數 (IK 求解)")
    parser.add_argument("input", help="輸入的 .npy joints 檔案 (T, 24, 3)")
    parser.add_argument("--output", help="輸出的 .pkl 檔案路徑")
    parser.add_argument("--gender", default="neutral", choices=["male", "female", "neutral"],
                        help="SMPL 性別 (預設: neutral)")
    
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
    
    # 2. Initialize IK Solver
    solver = SMPLIKSolver(gender=args.gender)
    
    # 3. Perform IK
    result = solver.fit_sequence(target_joints)
    
    # 4. Save result
    out_path = args.output
    if not out_path:
        os.makedirs("data/smpl/smpl", exist_ok=True)
        filename = os.path.basename(args.input).replace('.npy', '_fitted_smpl.pkl')
        out_path = os.path.join("data/smpl/smpl", filename)
        
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
        
    print(f"✅ Fitting complete! Result saved to: {out_path}")

if __name__ == "__main__":
    main()
