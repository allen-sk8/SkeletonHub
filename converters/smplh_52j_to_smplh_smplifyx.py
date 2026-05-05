import os
import sys
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

# 將專案路徑加入 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smpl.handler import SMPLHandler

# --- VPoser v1.0 模型定義 (直接嵌入以確保相容性) ---

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()
    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)

class VPoserV1(nn.Module):
    def __init__(self, num_neurons=512, latentD=32, data_shape=[1, 21, 3], use_cont_repr=True):
        super(VPoserV1, self).__init__()
        self.latentD = latentD
        n_features = np.prod(data_shape)
        self.num_joints = data_shape[1]
        self.bodyprior_enc_bn1 = nn.BatchNorm1d(n_features)
        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        self.dropout = nn.Dropout(p=.1)
        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)
        self.rot_decoder = ContinousRotReprDecoder() if use_cont_repr else None
        self.bodyprior_dec_out = nn.Linear(num_neurons, self.num_joints * 6)

    def decode(self, Zin, output_type='aa'):
        Xout = F.leaky_relu(self.bodyprior_dec_fc1(Zin), negative_slope=.2)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_dec_fc2(Xout), negative_slope=.2)
        Xout = self.bodyprior_dec_out(Xout)
        if self.rot_decoder:
            Xout = self.rot_decoder(Xout)
        else:
            Xout = torch.tanh(Xout)
        Xout = Xout.view([-1, 1, self.num_joints, 3, 3])
        if output_type == 'aa':
            return self.matrot2aa(Xout).view(Zin.shape[0], -1)
        return Xout

    @staticmethod
    def matrot2aa(pose_matrot):
        try:
            from pytorch3d.transforms import matrix_to_axis_angle
        except ImportError:
            raise ImportError("需要 pytorch3d 才能執行 VPoser 解碼。請執行 pip install pytorch3d")
        
        batch_size = pose_matrot.size(0)
        # pose_matrot 的 shape 是 (-1, 3, 3)
        # matrix_to_axis_angle 接受 (*, 3, 3) 並且回傳 (*, 3)
        pose = matrix_to_axis_angle(pose_matrot.view(-1, 3, 3)).view(batch_size, -1)
        return pose

# --- Wrapper 實作 ---

class SMPLifyX3DWrapper:
    def __init__(self, model_root="common_models/body_models", vposer_path="common_models/vposer_v1_0", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.handler = SMPLHandler(model_root=model_root, model_type="smplh")
        
        # 載入 VPoser v1.0
        self.vposer = None
        ckpt_path = os.path.join(vposer_path, "snapshots/TR00_E096.pt")
        if os.path.exists(ckpt_path):
            print(f"🚀 正在載入 VPoser v1.0 權重從 {ckpt_path}...")
            self.vposer = VPoserV1().to(self.device)
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.vposer.load_state_dict(state_dict)
            self.vposer.eval()
        else:
            print(f"⚠️ 找不到 VPoser 權重 ({ckpt_path})，將退回基礎 L2 優化。")

    def fit(self, target_joints, gender='neutral', num_iters=1000):
        T = target_joints.shape[0]
        num_joints = target_joints.shape[1]
        target_torch = torch.from_numpy(target_joints).to(self.device).float()
        model = self.handler._get_model(gender)
        
        # 優化參數
        transl = target_torch[:, 0, :].clone().detach().requires_grad_(True)
        global_orient = torch.zeros((T, 3), device=self.device, requires_grad=True)
        
        if self.vposer is not None:
            pose_embedding = torch.zeros((T, 32), device=self.device, requires_grad=True)
            body_pose = None
        else:
            pose_embedding = None
            body_pose = torch.zeros((T, 63), device=self.device, requires_grad=True)
            
        left_hand_pose = torch.zeros((T, 45), device=self.device, requires_grad=True)
        right_hand_pose = torch.zeros((T, 45), device=self.device, requires_grad=True)
        betas = torch.zeros((1, 10), device=self.device, requires_grad=True)
        
        print(f"🚀 [SMPLify-X 3D] 開始擬合 {T} 幀數據...")
        opt = torch.optim.Adam([transl, global_orient] + ([pose_embedding] if pose_embedding is not None else [body_pose]) + [left_hand_pose, right_hand_pose, betas], lr=0.01)
        
        for i in tqdm(range(num_iters)):
            opt.zero_grad()
            
            if self.vposer is not None:
                current_body_pose = self.vposer.decode(pose_embedding, output_type='aa')
            else:
                current_body_pose = body_pose
                
            output = model(
                betas=betas.repeat(T, 1),
                global_orient=global_orient,
                body_pose=current_body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=transl,
                return_verts=False
            )
            
            pred_joints = output.joints[:, :num_joints, :]
            loss_data = torch.mean((pred_joints - target_torch)**2)
            
            # Priors
            if self.vposer is not None:
                loss_prior = torch.mean(pose_embedding**2) * 0.01
            else:
                loss_prior = torch.mean(current_body_pose**2) * 0.01
                
            loss_prior += torch.mean(left_hand_pose**2) * 0.01 + \
                          torch.mean(right_hand_pose**2) * 0.01 + \
                          torch.mean(betas**2) * 0.1
                          
            total_loss = loss_data * 100.0 + loss_prior
            total_loss.backward()
            opt.step()
            
        final_body_pose = self.vposer.decode(pose_embedding, output_type='aa').detach() if self.vposer is not None else body_pose.detach()
        final_poses = torch.cat([global_orient, final_body_pose, left_hand_pose, right_hand_pose], dim=-1).detach().cpu().numpy()
        
        return {
            'poses': final_poses,
            'betas': betas.detach().cpu().numpy().squeeze(),
            'trans': transl.detach().cpu().numpy(),
            'gender': gender
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--gender", default='neutral')
    parser.add_argument("--output")
    args = parser.parse_args()

    data = np.load(args.input)
    wrapper = SMPLifyX3DWrapper()
    res = wrapper.fit(data, gender=args.gender)
    
    out = args.output or args.input.replace('.npy', '_smplifyx.pkl')
    with open(out, 'wb') as f:
        pickle.dump(res, f)
    print(f"✅ Saved to {out}")

if __name__ == "__main__":
    main()
