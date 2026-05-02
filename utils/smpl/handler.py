import os
import torch
import numpy as np
import smplx

class SMPLHandler:
    def __init__(self, model_root="common_models/body_models", model_type="smplh"):
        """
        處理 SMPL 家族模型的加載與數據計算
        """
        self.model_root = model_root
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def _get_model(self, gender):
        gender = str(gender).replace("b'", "").replace("'", "").lower()
        if gender not in ['male', 'female', 'neutral']: gender = 'neutral'
        
        if gender not in self.models:
            # 🌟 直接指向整理後的模型根目錄
            # 當 model_type="smplh" 時，smplx 庫會自動去 self.model_root/smplh/ 下尋找
            # 你已經把 SMPLH_FEMALE.pkl 放在 smplh/ 目錄下，這完全符合庫的自動搜索邏輯
            
            try:
                self.models[gender] = smplx.create(
                    model_path=self.model_root,
                    model_type=self.model_type,
                    gender=gender,
                    use_pca=False,
                    flat_hand_mean=True,
                    batch_size=1,
                    ext='pkl' # 優先使用你準備好的 .pkl 版本
                ).to(self.device)
                print(f"✅ 已從 {os.path.join(self.model_root, self.model_type)} 加載 {self.model_type} ({gender}) 模型")
            except Exception as e:
                print(f"❌ 模型加載失敗: {e}")
                raise e
            
        return self.models[gender]

    def params_to_vertices(self, poses, betas, trans, gender):
        """
        將參數轉換為 3D 頂點
        """
        model = self._get_model(gender)
        T = poses.shape[0]
        
        poses_torch = torch.from_numpy(poses).to(self.device).float()
        betas_torch = torch.from_numpy(betas).to(self.device).float()
        trans_torch = torch.from_numpy(trans).to(self.device).float()

        all_vertices = []
        batch_size = 50
        
        with torch.no_grad():
            for i in range(0, T, batch_size):
                end = min(i + batch_size, T)
                curr_T = end - i
                
                num_betas = model.num_betas
                curr_betas = betas_torch[:num_betas].unsqueeze(0).repeat(curr_T, 1)
                
                if self.model_type == 'smplh' and poses_torch.shape[1] >= 156:
                    output = model(
                        betas=curr_betas,
                        global_orient=poses_torch[i:end, :3],
                        body_pose=poses_torch[i:end, 3:66],
                        left_hand_pose=poses_torch[i:end, 66:111],
                        right_hand_pose=poses_torch[i:end, 111:156],
                        transl=trans_torch[i:end],
                        return_verts=True
                    )
                else:
                    output = model(
                        betas=curr_betas,
                        global_orient=poses_torch[i:end, :3],
                        body_pose=poses_torch[i:end, 3:72],
                        transl=trans_torch[i:end],
                        return_verts=True
                    )
                all_vertices.append(output.vertices.cpu().numpy())

        vertices = np.concatenate(all_vertices, axis=0)
        return vertices, model.faces

    def params_to_joints(self, poses, betas, trans, gender):
        """
        將參數轉換為 3D 關節座標 (Joints)
        """
        model = self._get_model(gender)
        T = poses.shape[0]
        
        poses_torch = torch.from_numpy(poses).to(self.device).float()
        betas_torch = torch.from_numpy(betas).to(self.device).float()
        trans_torch = torch.from_numpy(trans).to(self.device).float()

        all_joints = []
        batch_size = 50
        
        with torch.no_grad():
            for i in range(0, T, batch_size):
                end = min(i + batch_size, T)
                curr_T = end - i
                
                num_betas = model.num_betas
                curr_betas = betas_torch[:num_betas].unsqueeze(0).repeat(curr_T, 1)
                
                if self.model_type == 'smplh' and poses_torch.shape[1] >= 156:
                    output = model(
                        betas=curr_betas,
                        global_orient=poses_torch[i:end, :3],
                        body_pose=poses_torch[i:end, 3:66],
                        left_hand_pose=poses_torch[i:end, 66:111],
                        right_hand_pose=poses_torch[i:end, 111:156],
                        transl=trans_torch[i:end],
                        return_verts=False
                    )
                else:
                    output = model(
                        betas=curr_betas,
                        global_orient=poses_torch[i:end, :3],
                        body_pose=poses_torch[i:end, 3:72],
                        transl=trans_torch[i:end],
                        return_verts=False
                    )
                all_joints.append(output.joints.cpu().numpy())

        joints = np.concatenate(all_joints, axis=0)
        return joints
