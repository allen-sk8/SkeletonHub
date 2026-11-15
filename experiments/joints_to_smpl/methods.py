"""第一輪比較的五個版本。全部吃 (T,24,3) 的 Y-up 公尺關節，回傳
dict(poses (T,72), betas (10,), trans (T,3), gender)。現有轉換程式都不修改，只在這裡包裝或複製後改。
"""
import io
import os
import sys
import contextlib
import importlib.util
import numpy as np
import torch

from common import (ROOT, DEVICE, get_model, rest_joints, fit_betas_from_bones, init_global_orient,
                    axis_angle_to_matrix, matrix_to_axis_angle, SMPL_PARENTS)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------- 1. 現有版本原樣 ----------
def easymocap_orig(target):
    from utils.smpl import easymocap_wrapper as W
    w = _quiet(W.EasyMocapWrapper, model_type='smpl', device=DEVICE)
    r = _quiet(w.fit_3d, target.copy())
    return {'poses': r['poses'][:, :72].astype(np.float32),
            'betas': np.asarray(r['betas'], dtype=np.float32).reshape(-1)[:10],
            'trans': r['trans'].astype(np.float32), 'gender': 'neutral'}


# ---------- 2. 修正關節對應的 EasyMocap 版本 ----------
def easymocap_fixed(target, fix_trans=True):
    """修正一：模型輸出改成 SMPL 自己的 24 關節（原本是 BODY25 的 25 點）。
    修正二（fix_trans）：EasyMocap 的整體旋轉是繞原點轉，本專案其餘程式（smplx）是繞骨盆轉，
    兩者的位移差 R·J0 − J0，原本沒有換算。"""
    from utils.smpl import easymocap_wrapper as W

    class FixedModelWrapper(W.SMPLModelWrapper):
        def forward(self, *args, **kwargs):
            kwargs.setdefault('return_smpl_joints', True)
            return self.model(*args, **kwargs)

    w = _quiet(W.EasyMocapWrapper, model_type='smpl', device=DEVICE)
    w.body_model = FixedModelWrapper(w.body_model.model)
    r = _quiet(w.fit_3d, target.copy())
    poses = r['poses'][:, :72].astype(np.float32)
    trans = r['trans'].astype(np.float32)
    betas = np.asarray(r['betas'], dtype=np.float32).reshape(-1)[:10]
    if fix_trans:
        J0 = rest_joints(torch.as_tensor(betas)).detach()[0]
        R = axis_angle_to_matrix(torch.as_tensor(poses[:, :3]))
        trans = trans + (R @ J0 - J0).numpy()
    return {'poses': poses, 'betas': betas, 'trans': trans, 'gender': 'neutral'}


def easymocap_fixed_order_only(target):
    return easymocap_fixed(target, fix_trans=False)


# ---------- 3. repo 裡手寫的擬合程式（原樣） ----------
def handmade(target):
    from utils.smpl.smpl_ik_solver import SMPLIKSolver
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        s = _quiet(SMPLIKSolver, gender='neutral', device=DEVICE)
        r = _quiet(s.fit_sequence, target.copy(), verbose=False)
    finally:
        os.chdir(cwd)
    return {'poses': r['poses'], 'betas': r['betas'].reshape(-1), 'trans': r['trans'], 'gender': 'neutral'}


# ---------- 4. HybrIK 解析式解法 ----------
_hybrik_lbs = None


def _load_hybrik_lbs():
    global _hybrik_lbs
    if _hybrik_lbs is None:
        path = os.path.join(ROOT, 'external', 'Hybrik', 'hybrik', 'models', 'layers', 'smpl', 'lbs.py')
        spec = importlib.util.spec_from_file_location('hybrik_lbs', path)
        _hybrik_lbs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_hybrik_lbs)
    return _hybrik_lbs


def hybrik_analytic(target, spine3_three_children=True):
    """直接呼叫 HybrIK 的 batch_inverse_kinematics_transform。
    - 體型：由輸入骨長估（中性模型）。
    - 軸向扭轉（phis）：沒有影像網路可預測，全部設為零（cos=1, sin=0）。
    - 5 個沒有子關節的末端（腳掌 10、11，頭 15，手掌 22、23）：給單位矩陣。
    - 上胸（9 號）有脖子與兩個鎖骨三個子關節：用函式內建的三子關節 SVD 分支。
      HybrIK 自己的 24 點設定只看脖子一個子關節再靠預測的扭轉補，這裡沒有扭轉可用。
    """
    lbs = _load_hybrik_lbs()
    T = target.shape[0]
    tgt = torch.as_tensor(target, dtype=torch.float32)
    betas = fit_betas_from_bones(target)
    J = rest_joints(betas).detach()
    parents = torch.tensor(SMPL_PARENTS, dtype=torch.long)
    children = torch.full((24,), -1, dtype=torch.long)
    for i in range(1, 24):
        if children[parents[i]] < 0:
            children[parents[i]] = i
    children[0] = 3
    children[9] = -3 if spine3_three_children else 12
    phis = torch.zeros(T, 23, 2)
    phis[..., 0] = 1.0
    leaf = torch.eye(3).repeat(T, 5, 1, 1)
    rot_mats, _ = lbs.batch_inverse_kinematics_transform(
        tgt - tgt[:, :1], None, phis, J[None].expand(T, -1, -1).clone(),
        children, parents, dtype=torch.float32, train=False, leaf_thetas=leaf)
    poses = matrix_to_axis_angle(rot_mats).reshape(T, 72)
    trans = tgt[:, 0] - J[0]
    return {'poses': poses.numpy(), 'betas': betas.numpy(), 'trans': trans.numpy(), 'gender': 'neutral'}


# ---------- 5. VPoser（第 1 版權重）加最佳化 ----------
_vposer = None


def _load_vposer():
    global _vposer
    if _vposer is None:
        path = os.path.join(ROOT, 'converters', 'joints_52j_to_smplh_smplifyx.py')
        spec = importlib.util.spec_from_file_location('smplifyx_conv', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        vp = mod.VPoserV1()
        ckpt = os.path.join(ROOT, 'common_models', 'vposer_v1_0', 'snapshots', 'TR00_E096.pt')
        vp.load_state_dict(torch.load(ckpt, map_location='cpu'))
        vp.eval()
        for p in vp.parameters():
            p.requires_grad_(False)
        _vposer = vp
    return _vposer


def vposer_opt(target, iters=(150, 500), w_prior=2e-4, w_smooth=2e-3, w_beta=1e-4,
               w_hand=1.0, w_wrist=0.0, w_wrist_smooth=0.0):
    """身體 21 個關節的姿態不直接調，而是調 VPoser 的 32 維潛在向量再解碼，所以不容易出現不像人的姿態。
    起點：體型由骨長估；整體朝向由軀幹關節對齊估；潛在向量為零（VPoser 的平均姿態）。
    手掌 22、23 固定為零（24 個關節點看不出手掌旋轉）。"""
    vp = _load_vposer()
    model = get_model('neutral')
    T = target.shape[0]
    tgt = torch.as_tensor(target, dtype=torch.float32)
    betas = fit_betas_from_bones(target).clone().requires_grad_(True)
    J = rest_joints(betas.detach()).detach()
    orient = init_global_orient(target, J).clone().requires_grad_(True)
    transl = (tgt[:, 0] - J[0]).clone().requires_grad_(True)
    z = torch.zeros(T, 32, requires_grad=True)
    hands = torch.zeros(T, 6)
    jw = torch.ones(24)
    jw[22:] = w_hand                                 # 手掌點（22、23）的權重

    def forward():
        body = vp.decode(z, output_type='aa').reshape(T, 63)
        out = model(betas=betas[None].expand(T, -1), global_orient=orient,
                    body_pose=torch.cat([body, hands], 1), transl=transl, return_verts=True)
        return out.joints[:, :24], body

    def run(params, n, lr):
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(n):
            opt.zero_grad()
            pj, body = forward()
            loss = (((pj - tgt) ** 2).sum(-1) * jw).mean()
            if w_wrist > 0 or w_wrist_smooth > 0:
                wrist = body[:, 57:63]                      # 手腕 20、21 的軸角
                loss = loss + w_wrist * (wrist ** 2).mean()
                if T > 1:
                    loss = loss + w_wrist_smooth * ((wrist[1:] - wrist[:-1]) ** 2).mean()
            loss = loss + w_prior * (z ** 2).mean() + w_beta * (betas ** 2).mean()
            if T > 1:
                loss = loss + w_smooth * ((z[1:] - z[:-1]) ** 2).mean()
            loss.backward()
            opt.step()

    run([orient, transl], iters[0], 0.02)
    run([orient, transl, z, betas], iters[1], 0.05)
    with torch.no_grad():
        _, body = forward()
    poses = torch.cat([orient, body, hands], 1).detach().numpy()
    return {'poses': poses, 'betas': betas.detach().numpy(), 'trans': transl.detach().numpy(), 'gender': 'neutral'}


def vposer_opt_wristfix(target):
    """手腕修正版：偵測器給的手掌點（離手腕只有約 8.5 公分）很抖，小小的位置誤差就會變成很大的手腕角度。
    做法：手掌點的權重降到 0.05；手腕角度加一個往零拉的懲罰；手腕相鄰幀的變化加懲罰。"""
    return vposer_opt(target, w_hand=0.05, w_wrist=2e-3, w_wrist_smooth=2e-2)


def to_smplh_relaxed(pred, target):
    """把擬合出來的 SMPL 身體姿態搬到 SMPL-H，手指用 MANO 的平均放鬆手型（SMPL 的手指不能彎，永遠是平手）。
    SMPL-H 的體型空間和 SMPL 不同，所以體型重新由骨長估（只用身體 22 個關節）。
    回傳 (dict(poses (T,156), betas, trans, gender='neutral'), 頂點 (T,6890,3))。手部姿態是明確寫出的數值，
    對應本專案 SMPLHandler 的 flat_hand_mean=True 慣例。"""
    import smplx
    from common import MODEL_ROOT, BONES
    mh = smplx.create(MODEL_ROOT, model_type='smplh', gender='neutral', use_pca=False, flat_hand_mean=True,
                      ext='npz', batch_size=1)
    mm = smplx.create(MODEL_ROOT, model_type='smplh', gender='male', use_pca=False, flat_hand_mean=False,
                      ext='pkl', batch_size=1)
    lh, rh = mm.left_hand_mean.detach().clone(), mm.right_hand_mean.detach().clone()
    tgt = torch.as_tensor(target, dtype=torch.float32)
    bones = [(p, c) for p, c in BONES if c <= 21]
    ip, ic = [p for p, c in bones], [c for p, c in bones]
    tgt_len = (tgt[:, ic] - tgt[:, ip]).norm(dim=-1).median(dim=0).values
    betas = torch.zeros(10, requires_grad=True)
    opt = torch.optim.LBFGS([betas], lr=1.0, max_iter=100, line_search_fn='strong_wolfe')

    def rest(b):
        v = mh.v_template + torch.einsum('vck,k->vc', mh.shapedirs[:, :, :10], b)
        return mh.J_regressor @ v

    def closure():
        opt.zero_grad()
        J = rest(betas)
        loss = (((J[ic] - J[ip]).norm(dim=-1) - tgt_len) ** 2).sum() + 1e-5 * (betas ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    betas = betas.detach().clamp(-5, 5)
    T = tgt.shape[0]
    poses = torch.as_tensor(pred['poses'], dtype=torch.float32)
    with torch.no_grad():
        pelvis = get_model('neutral')(betas=torch.as_tensor(pred['betas'], dtype=torch.float32).reshape(1, -1).expand(T, -1),
                                      global_orient=poses[:, :3], body_pose=poses[:, 3:72],
                                      transl=torch.as_tensor(pred['trans'], dtype=torch.float32)).joints[:, 0]
        trans = pelvis - rest(betas)[0]
        out = mh(betas=betas[None].expand(T, -1), global_orient=poses[:, :3], body_pose=poses[:, 3:66],
                 left_hand_pose=lh[None].expand(T, -1), right_hand_pose=rh[None].expand(T, -1), transl=trans)
    full = torch.cat([poses[:, :66], lh[None].expand(T, -1), rh[None].expand(T, -1)], 1)
    return ({'poses': full.numpy(), 'betas': betas.numpy(), 'trans': trans.numpy(), 'gender': 'neutral'},
            out.vertices.numpy())


H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
H36M_BONES = [(H36M_PARENTS[i], i) for i in range(1, 17)]
_J17 = None


def h36m_regressor():
    """H36M 17 點的回歸矩陣 (17, 6890)：由 SMPL 網格頂點線性組合出 17 個點。"""
    global _J17
    if _J17 is None:
        _J17 = torch.as_tensor(np.load(os.path.join(ROOT, 'common_models', 'body_models', 'smpl',
                                                    'J_regressor_h36m.npy')), dtype=torch.float32)
    return _J17


def normalize_h36m17(joints):
    """FineFS 官方 17 點的前處理（會改動輸入，屬於「轉換」而不是「擬合」）：
    1. 以骨盆為原點（官方骨架每幀把最低點貼在地面、骨盆水平位置固定為零，本來就沒有整體位移）。
    2. 每一幀各自縮放，讓 16 段骨頭的總長等於整段的中位數（官方骨架同一段內骨長會變動）。
    3. 整段再縮放一次，讓骨頭總長等於平均體型 SMPL 的對應總長（丟掉絕對身高）。
    回傳 (處理後的關節, 每幀縮放倍率, 整段縮放倍率)。"""
    j = np.asarray(joints, dtype=np.float32)
    j = j - j[:, :1]
    ip, ic = [p for p, c in H36M_BONES], [c for p, c in H36M_BONES]
    tot = np.linalg.norm(j[:, ic] - j[:, ip], axis=-1).sum(1)
    per_frame = np.median(tot) / tot
    j = j * per_frame[:, None, None]
    with torch.no_grad():
        m = get_model('neutral')
        r = h36m_regressor() @ m.v_template
        ref = float((r[ic] - r[ip]).norm(dim=-1).sum())
    g = ref / float(np.median(tot))
    return (j * g).astype(np.float32), per_frame, g


def vposer_opt_h36m17(target17, iters=(150, 500), w_prior=2e-4, w_smooth=2e-3, w_beta=1e-3, w_wrist=2e-3,
                      w_wrist_smooth=2e-2):
    """吃 H36M 17 點（Y-up、公尺、骨盆在原點）的 VPoser 加最佳化。
    模型這邊的 17 點是用回歸矩陣從 SMPL 網格算出來的。17 點沒有手掌、腳尖、鎖骨，
    所以手腕、腳掌的旋轉完全靠 VPoser 的先驗與「往零拉」的懲罰決定。"""
    vp = _load_vposer()
    model = get_model('neutral')
    J17 = h36m_regressor()
    T = target17.shape[0]
    tgt = torch.as_tensor(target17, dtype=torch.float32)
    betas = torch.zeros(10, requires_grad=True)
    with torch.no_grad():
        rest17 = J17 @ model.v_template
    torso = [0, 1, 4, 7, 8, 11, 14]
    from common import kabsch
    orient = matrix_to_axis_angle(kabsch(rest17[torso] - rest17[0], tgt[:, torso] - tgt[:, :1])).clone().requires_grad_(True)
    transl = (tgt[:, 0] - rest17[0]).clone().requires_grad_(True)
    z = torch.zeros(T, 32, requires_grad=True)
    hands = torch.zeros(T, 6)

    def forward():
        body = vp.decode(z, output_type='aa').reshape(T, 63)
        out = model(betas=betas[None].expand(T, -1), global_orient=orient,
                    body_pose=torch.cat([body, hands], 1), transl=transl, return_verts=True)
        return torch.einsum('jv,tvc->tjc', J17, out.vertices), body

    def run(params, n, lr):
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(n):
            opt.zero_grad()
            pj, body = forward()
            wrist = body[:, 57:63]
            loss = ((pj - tgt) ** 2).sum(-1).mean() + w_prior * (z ** 2).mean() + w_beta * (betas ** 2).mean()
            loss = loss + w_wrist * (wrist ** 2).mean()
            if T > 1:
                loss = loss + w_smooth * ((z[1:] - z[:-1]) ** 2).mean() + w_wrist_smooth * ((wrist[1:] - wrist[:-1]) ** 2).mean()
            loss.backward()
            opt.step()

    run([orient, transl], iters[0], 0.02)
    run([orient, transl, z, betas], iters[1], 0.05)
    with torch.no_grad():
        _, body = forward()
    poses = torch.cat([orient, body, hands], 1).detach().numpy()
    return {'poses': poses, 'betas': betas.detach().numpy(), 'trans': transl.detach().numpy(), 'gender': 'neutral'}


def regress_h36m17(pred):
    """由擬合結果算出模型這邊的 17 點 (T,17,3)。"""
    from common import fk
    with torch.no_grad():
        v = fk(pred['poses'], pred['betas'], pred['trans'], pred['gender'], verts=True)[1]
        return torch.einsum('jv,tvc->tjc', h36m_regressor(), v).numpy()


METHODS = {
    'easymocap_orig': easymocap_orig,
    'easymocap_fixed_order_only': easymocap_fixed_order_only,
    'easymocap_fixed': easymocap_fixed,
    'handmade': handmade,
    'hybrik_analytic': hybrik_analytic,
    'vposer_opt': vposer_opt,
    'vposer_opt_wristfix': vposer_opt_wristfix,
    'vposer_opt_h36m17': vposer_opt_h36m17,
}
