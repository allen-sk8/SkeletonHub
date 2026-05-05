import os
import sys
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm

# Example Usage:
# python converters/joints_to_smplh.py data/smpl_joints/samples_22j/012314.npy --axis Y --output data/smpl/smplh/012314_fitted.pkl

# 將專案路徑加入以匯入工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.handler import SMPLHandler

class JointsToSMPLHFitter:
    def __init__(self, model_root="common_models/body_models", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.handler = SMPLHandler(model_root=model_root, model_type="smplh")
        
    def fit(self, target_joints, gender='neutral', num_iters=1000, lr=1e-2):
        """
        將 3D 關節座標擬合至 SMPL-H 參數 (支援 52 關節，含手指)
        target_joints: (T, 52, 3) (必須為 Y-up)
        """
        T = target_joints.shape[0]
        num_joints = target_joints.shape[1]
        
        if num_joints != 52:
            print(f"⚠️ 警告: 輸入關節數為 {num_joints}，而非預期的 52 (SMPL-H 標準)。將嘗試匹配前 {min(num_joints, 52)} 個關節。")

        target_torch = torch.from_numpy(target_joints).to(self.device).float()
        
        # 2. 初始化待優化參數
        global_orient = torch.zeros((T, 3), device=self.device, requires_grad=True)
        body_pose = torch.zeros((T, 63), device=self.device, requires_grad=True)
        left_hand_pose = torch.zeros((T, 45), device=self.device, requires_grad=True)
        right_hand_pose = torch.zeros((T, 45), device=self.device, requires_grad=True)
        betas = torch.zeros((1, 16), device=self.device, requires_grad=True)
        transl = torch.zeros((T, 3), device=self.device, requires_grad=True)
        
        # 簡單的平移初始化
        with torch.no_grad():
            transl += target_torch[:, 0, :]
            
        model = self.handler._get_model(gender)
        num_betas = model.num_betas

        # 3. 分階段優化 (Staged Optimization)
        # 階段一：只優化全域位移與朝向 (Global)
        print("🚀 [Stage 1] 優化全域位移與朝向...")
        opt_global = torch.optim.Adam([transl, global_orient], lr=lr*2)
        for i in range(200):
            opt_global.zero_grad()
            output = model(
                betas=betas[:, :num_betas].repeat(T, 1),
                global_orient=global_orient,
                body_pose=torch.zeros((T, 63), device=self.device),
                left_hand_pose=torch.zeros((T, 45), device=self.device),
                right_hand_pose=torch.zeros((T, 45), device=self.device),
                transl=transl,
                return_verts=False
            )
            loss = torch.mean((output.joints[:, :num_joints, :] - target_torch)**2)
            loss.backward()
            opt_global.step()

        # 階段二：加入身體姿勢與體型 (Body + Betas)
        print("🚀 [Stage 2] 優化身體姿勢與體型...")
        opt_body = torch.optim.Adam([transl, global_orient, body_pose, betas], lr=lr)
        for i in range(400):
            opt_body.zero_grad()
            output = model(
                betas=betas[:, :num_betas].repeat(T, 1),
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=torch.zeros((T, 45), device=self.device),
                right_hand_pose=torch.zeros((T, 45), device=self.device),
                transl=transl,
                return_verts=False
            )
            loss_data = torch.mean((output.joints[:, :num_joints, :] - target_torch)**2)
            loss_prior = torch.mean(body_pose**2) * 0.01 + torch.mean(betas**2) * 0.1
            (loss_data + loss_prior).backward()
            opt_body.step()

        # 階段三：加入手部細節 (Full SMPL-H)
        print("🚀 [Stage 3] 全面優化 (含手部細節)...")
        opt_full = torch.optim.Adam([transl, global_orient, body_pose, left_hand_pose, right_hand_pose, betas], lr=lr*0.5)
        for i in tqdm(range(num_iters)):
            opt_full.zero_grad()
            output = model(
                betas=betas[:, :num_betas].repeat(T, 1),
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=transl,
                return_verts=False
            )
            pred_joints = output.joints[:, :num_joints, :]
            
            loss_data = torch.mean((pred_joints - target_torch)**2)
            # 增加手部與身體的 Prior 懲罰，避免過度扭曲
            loss_prior = torch.mean(body_pose**2) * 0.01 + \
                         torch.mean(left_hand_pose**2) * 0.01 + \
                         torch.mean(right_hand_pose**2) * 0.01 + \
                         torch.mean(betas**2) * 0.1
            
            total_loss = loss_data + loss_prior
            total_loss.backward()
            opt_full.step()
            
            if (i+1) % 200 == 0:
                tqdm.write(f"Iteration {i+1}/{num_iters}, Loss: {total_loss.item():.6f}")

        # 4. 打包結果
        final_poses = torch.cat([global_orient, body_pose, left_hand_pose, right_hand_pose], dim=-1).detach().cpu().numpy()
        final_betas = betas.detach().cpu().numpy().squeeze()
        final_trans = transl.detach().cpu().numpy()
        
        result = {
            'poses': final_poses,
            'betas': final_betas,
            'trans': final_trans,
            'gender': gender,
            'num_joints': num_joints
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(description="SMPL-H 52j to SMPL-H (.pkl) 擬合轉換器")
    parser.add_argument("input", help="輸入的 .npy 關節路徑 (T, 52, 3)")
    parser.add_argument("--gender", choices=['male', 'female', 'neutral'], default='neutral')
    parser.add_argument("--iters", type=int, default=1000, help="最終階段優化疊代次數")
    parser.add_argument("--output", help="輸出 .pkl 路徑")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 找不到檔案: {args.input}")
        return

    # 執行擬合
    fitter = JointsToSMPLHFitter()
    target_joints = np.load(args.input)
    
    result = fitter.fit(
        target_joints=target_joints,
        gender=args.gender,
        num_iters=args.iters
    )
    
    # 儲存
    output_path = args.output
    if not output_path:
        os.makedirs("data/smpl/smplh", exist_ok=True)
        output_path = os.path.join("data/smpl/smplh", os.path.basename(args.input).replace('.npy', '_52j_fitted.pkl'))
        
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
        
    print(f"✅ 擬合完成！結果已儲存至: {output_path}")

if __name__ == "__main__":
    main()
