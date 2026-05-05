import os
import torch
import numpy as np
import smplx

class SMPLHandler:
    def __init__(self, model_root="common_models/body_models", model_type="smplh"):
        """
        處理 SMPL 家族模型的加載與數據計算
        """
        self.model_root = os.path.abspath(model_root)
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def _get_model(self, gender):
        gender = str(gender).replace("b'", "").replace("'", "").lower()
        if gender not in ['male', 'female', 'neutral']: gender = 'neutral'
        
        if gender not in self.models:
            # 🌟 自動嘗試不同副檔名與性別回溯
            extensions = ['pkl', 'npz']
            success = False
            
            # 若為 neutral 且找不到，會嘗試 female/male
            genders_to_try = [gender]
            if gender == 'neutral':
                genders_to_try += ['female', 'male']
                
            for g in genders_to_try:
                for ext in extensions:
                    try:
                        self.models[gender] = smplx.create(
                            model_path=self.model_root,
                            model_type=self.model_type,
                            gender=g,
                            use_pca=False,
                            flat_hand_mean=True,
                            batch_size=1,
                            ext=ext
                        ).to(self.device)
                        print(f"✅ 已加載 {self.model_type} (性別: {g}, 格式: {ext})")
                        success = True
                        break
                    except:
                        continue
                if success: break
            
            if not success:
                print(f"❌ 模型加載失敗: 無法在 {self.model_root} 找到支援的 {self.model_type} 模型檔案")
                raise FileNotFoundError
            
        return self.models[gender]

    def params_to_vertices(self, poses, betas, trans, gender):
        """
        將參數轉換為 3D 頂點
        """
        model = self._get_model(gender)
        
        # 🌟 確保輸入至少是 2D (T, D)
        if poses.ndim == 1: poses = poses[None, ...]
        if trans.ndim == 1: trans = trans[None, ...]
        if betas.ndim == 1: betas = betas[None, ...]
        
        T = poses.shape[0]
        # 若 trans 只有 1 幀但 poses 有多幀，則自動擴展 trans
        if trans.shape[0] == 1 and T > 1:
            trans = np.repeat(trans, T, axis=0)
            
        poses_torch = torch.from_numpy(poses).to(self.device).float()
        trans_torch = torch.from_numpy(trans).to(self.device).float()
        betas_torch = torch.from_numpy(betas).to(self.device).float()
        
        all_vertices = []
        batch_size = 50
        
        # 🌟 初始化基礎 betas (1, num_betas)
        num_betas = model.num_betas
        # betas_torch 可能來自於 (1, 10) 或 (T, 10)，我們取第一幀作為基礎
        curr_betas = betas_torch[0:1, :num_betas]
        
        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            curr_batch_size = end - i
            
            # 🌟 顯式擴展 betas 以匹配當前 batch 大小，防止 smplx 內部廣播報錯
            batch_betas = curr_betas.expand(curr_batch_size, -1)
            
            with torch.no_grad():
                if self.model_type == 'smplh' and poses_torch.shape[1] >= 156:
                    output = model(
                        betas=batch_betas,
                        global_orient=poses_torch[i:end, :3],
                        body_pose=poses_torch[i:end, 3:66],
                        left_hand_pose=poses_torch[i:end, 66:111],
                        right_hand_pose=poses_torch[i:end, 111:156],
                        transl=trans_torch[i:end],
                        return_verts=True
                    )
                else:
                    output = model(
                        betas=batch_betas,
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
        將參數轉換為 3D 關節點 (J, 3)
        """
        model = self._get_model(gender)

        # 🌟 確保輸入至少是 2D (T, D)
        if poses.ndim == 1: poses = poses[None, ...]
        if trans.ndim == 1: trans = trans[None, ...]
        if betas.ndim == 1: betas = betas[None, ...]
        
        T = poses.shape[0]
        # 若 trans 只有 1 幀但 poses 有多幀，則自動擴展 trans
        if trans.shape[0] == 1 and T > 1:
            trans = np.repeat(trans, T, axis=0)

        poses_torch = torch.from_numpy(poses).to(self.device).float()
        trans_torch = torch.from_numpy(trans).to(self.device).float()
        betas_torch = torch.from_numpy(betas).to(self.device).float()

        all_joints = []
        batch_size = 50
        
        # 🌟 初始化基礎 betas (1, num_betas)
        num_betas = model.num_betas
        curr_betas = betas_torch[0:1, :num_betas]
        
        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            curr_batch_size = end - i
            
            # 🌟 顯式擴展 betas
            batch_betas = curr_betas.expand(curr_batch_size, -1)
            
            with torch.no_grad():
                if self.model_type == 'smplh' and poses_torch.shape[1] >= 156:
                    output = model(
                        betas=batch_betas,
                        global_orient=poses_torch[i:end, :3],
                        body_pose=poses_torch[i:end, 3:66],
                        left_hand_pose=poses_torch[i:end, 66:111],
                        right_hand_pose=poses_torch[i:end, 111:156],
                        transl=trans_torch[i:end],
                        return_verts=False,
                        return_joints=True
                    )
                else:
                    output = model(
                        betas=batch_betas,
                        global_orient=poses_torch[i:end, :3],
                        body_pose=poses_torch[i:end, 3:72],
                        transl=trans_torch[i:end],
                        return_verts=False,
                        return_joints=True
                    )
                all_joints.append(output.joints.cpu().numpy())

        joints = np.concatenate(all_joints, axis=0)
        return joints
