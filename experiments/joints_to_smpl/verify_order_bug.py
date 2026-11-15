"""驗證 joints_24j_to_smpl.py 的誤差是否來自關節順序對錯。
不改任何現有檔案：在記憶體裡把 EasyMocap 包裝的輸出從 BODY25 改成 SMPL 24 關節再跑一次。
用法（只用 CPU）： CUDA_VISIBLE_DEVICES="" python experiments/joints_to_smpl/verify_order_bug.py
"""
import os, sys, io, pickle, contextlib
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT); os.chdir(ROOT)
torch.set_num_threads(8)
from utils.smpl.handler import SMPLHandler
from utils.smpl import easymocap_wrapper as W

d = pickle.load(open('data/smpl/smpl/walking_01_poses_smpl.pkl', 'rb'))
h = SMPLHandler(model_type='smpl')
N = 40
target = h.params_to_joints(d['poses'][:N], d['betas'], d['trans'][:N], d['gender'])[:, :24]

def run(tag):
    w = W.EasyMocapWrapper(model_type='smpl')
    with contextlib.redirect_stdout(io.StringIO()):
        r = w.fit_3d(target.copy())
    betas = np.asarray(r['betas']).reshape(-1)[:10]
    pred = h.params_to_joints(r['poses'][:, :72].astype(np.float32), betas.astype(np.float32),
                              r['trans'].astype(np.float32), r['gender'])[:, :24]
    err = np.linalg.norm(pred - target, axis=-1)
    pa = np.linalg.norm((pred - pred[:, :1]) - (target - target[:, :1]), axis=-1)
    print(f"RESULT {tag}: 關節誤差 平均 {err.mean()*100:.1f} 公分, 最大 {err.max()*100:.1f} 公分; 骨盆對齊後平均 {pa.mean()*100:.1f} 公分")

run("修正前（原樣）")
orig_forward = W.SMPLModelWrapper.forward
def fwd(self, *a, **k):
    k.setdefault('return_smpl_joints', True)   # 改成輸出 SMPL 自己的 24 關節
    return self.model(*a, **k)
W.SMPLModelWrapper.forward = fwd
run("修正後（比對 SMPL 24 關節）")
