"""共用工具：SMPL 前向計算、由骨長估體型、指標。只用 CPU。"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.smpl.handler import SMPLHandler  # noqa: E402  (含 chumpy 相容修補)
import smplx  # noqa: E402
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402

MODEL_ROOT = os.path.join(ROOT, 'common_models', 'body_models')
DEVICE = torch.device('cpu')
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
BONES = [(SMPL_PARENTS[i], i) for i in range(1, 24)]
# 旋轉誤差分兩組：末端（沒有子關節，旋轉無法由關節點決定）與其餘。手掌 22、23 不計。
LEAF_JOINTS = [10, 11, 15]
MAIN_JOINTS = [j for j in range(22) if j not in LEAF_JOINTS]

_models = {}


def get_model(gender='neutral'):
    if gender not in _models:
        _models[gender] = smplx.create(MODEL_ROOT, model_type='smpl', gender=gender,
                                       batch_size=1).to(DEVICE)
    return _models[gender]


def fk(poses, betas, trans, gender='neutral', verts=False):
    """poses (T,72) 軸角、betas (10,) 或 (1,10)、trans (T,3) -> 關節 (T,24,3)[, 頂點 (T,6890,3)]"""
    m = get_model(gender)
    poses = torch.as_tensor(poses, dtype=torch.float32)
    trans = torch.as_tensor(trans, dtype=torch.float32)
    betas = torch.as_tensor(betas, dtype=torch.float32).reshape(1, -1)[:, :10]
    T = poses.shape[0]
    out = m(betas=betas.expand(T, -1), global_orient=poses[:, :3], body_pose=poses[:, 3:72],
            transl=trans, return_verts=True)
    if verts:
        return out.joints[:, :24], out.vertices
    return out.joints[:, :24]


def rest_joints(betas, gender='neutral'):
    """靜止姿態的 24 關節 (24,3)，對 betas 可微分。"""
    m = get_model(gender)
    betas = betas.reshape(-1)[:10]
    v = m.v_template + torch.einsum('vck,k->vc', m.shapedirs[:, :, :10], betas)
    return m.J_regressor @ v


def fit_betas_from_bones(target, gender='neutral', reg=1e-3):
    """由輸入關節的骨長（各幀取中位數）估體型。target: (T,24,3) ndarray。回傳 (10,) tensor。"""
    tgt = torch.as_tensor(target, dtype=torch.float32)
    idx_p = [p for p, c in BONES]
    idx_c = [c for p, c in BONES]
    tgt_len = (tgt[:, idx_c] - tgt[:, idx_p]).norm(dim=-1).median(dim=0).values
    betas = torch.zeros(10, requires_grad=True)
    opt = torch.optim.LBFGS([betas], lr=1.0, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        J = rest_joints(betas, gender)
        L = (J[idx_c] - J[idx_p]).norm(dim=-1)
        loss = ((L - tgt_len) ** 2).sum() + reg * (betas ** 2).sum() * 1e-2
        loss.backward()
        return loss
    opt.step(closure)
    return betas.detach().clamp(-5, 5)


def kabsch(A, B):
    """求 R 使 R @ A_i ≈ B_i。A: (N,3)，B: (T,N,3)，都已去中心。回傳 (T,3,3)。"""
    H = torch.einsum('ni,tnj->tij', A, B)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2))
    D = torch.eye(3).repeat(H.shape[0], 1, 1)
    D[:, 2, 2] = d
    return Vt.transpose(1, 2) @ D @ U.transpose(1, 2)


def init_global_orient(target, J_rest):
    """用軀幹關節（骨盆、髖、脊椎、鎖骨、肩）對齊靜止姿態，估整體朝向。回傳軸角 (T,3)。"""
    torso = [0, 1, 2, 3, 6, 9, 13, 14, 16, 17]
    tgt = torch.as_tensor(target, dtype=torch.float32)
    A = J_rest[torso] - J_rest[0]
    B = tgt[:, torso] - tgt[:, :1]
    return matrix_to_axis_angle(kabsch(A, B))


# ---------------- 指標 ----------------

def geodesic_deg(R1, R2):
    R = R1.transpose(-1, -2) @ R2
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1)) - 1) / 2
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))


def accel(x, fps):
    """關節加速度大小的平均（公尺/秒平方）。x: (T,J,3)"""
    a = (x[2:] - 2 * x[1:-1] + x[:-2]) * fps * fps
    return float(np.linalg.norm(a, axis=-1).mean())


def evaluate(pred, gt, fps=20.0):
    """pred: dict(poses, betas, trans, gender)；gt: 同格式，再加 'joints_clean'。全部回傳公釐或度。"""
    with torch.no_grad():
        pj, pv = fk(pred['poses'], pred['betas'], pred['trans'], pred['gender'], verts=True)
        gj, gv = fk(gt['poses'], gt['betas'], gt['trans'], gt['gender'], verts=True)
        Rp = axis_angle_to_matrix(torch.as_tensor(pred['poses'], dtype=torch.float32).reshape(-1, 24, 3))
        Rg = axis_angle_to_matrix(torch.as_tensor(gt['poses'], dtype=torch.float32).reshape(-1, 24, 3))
        ang = geodesic_deg(Rp, Rg)
        # 沿骨架累乘後的「整段肢體朝向」誤差，不受旋轉如何分配到各關節影響
        Gp, Gg = [Rp[:, 0]], [Rg[:, 0]]
        for j in range(1, 24):
            Gp.append(Gp[SMPL_PARENTS[j]] @ Rp[:, j])
            Gg.append(Gg[SMPL_PARENTS[j]] @ Rg[:, j])
        gang = geodesic_deg(torch.stack(Gp, 1), torch.stack(Gg, 1))
    pj, gj, pv, gv = pj.numpy(), gj.numpy(), pv.numpy(), gv.numpy()
    err = np.linalg.norm(pj - gj, axis=-1)
    err_root = np.linalg.norm((pj - pj[:, :1]) - (gj - gj[:, :1]), axis=-1)
    return {
        'joint_mm': float(err.mean() * 1000),
        'joint_max_mm': float(err.max() * 1000),
        'joint_rootaligned_mm': float(err_root.mean() * 1000),
        'rot_main_deg': float(ang[:, MAIN_JOINTS].mean()),
        'rot_leaf_deg': float(ang[:, LEAF_JOINTS].mean()),
        'limb_orient_main_deg': float(gang[:, MAIN_JOINTS].mean()),
        'vertex_mm': float(np.linalg.norm(pv - gv, axis=-1).mean() * 1000),
        'accel_pred': accel(pj, fps),
        'accel_gt': accel(gj, fps),
    }
