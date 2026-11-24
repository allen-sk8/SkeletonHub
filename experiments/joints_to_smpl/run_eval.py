"""第一輪比較的執行程式。只用 CPU。

AMASS（有標準答案）：
    CUDA_VISIBLE_DEVICES="" python experiments/joints_to_smpl/run_eval.py amass [--workers 6] [--frames 150]
滑冰（沒有標準答案）：
    CUDA_VISIBLE_DEVICES="" python experiments/joints_to_smpl/run_eval.py skating
結果寫在 experiments/joints_to_smpl/results/ 底下，每個（片段、輸入種類、方法）一個 json 和一個 pkl。
已經有 json 的組合會跳過，所以中斷後可以直接重跑。
"""
import os
import sys
import glob
import json
import time
import pickle
import argparse
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
THREADS = 4
os.environ.setdefault('OMP_NUM_THREADS', str(THREADS))
os.environ.setdefault('MKL_NUM_THREADS', str(THREADS))

RESULTS = os.environ.get('J2S_RESULTS', os.path.join(HERE, 'results'))
SKATING = {
    '2A+2Lo_0001_gvhmr': 'data/smpl_joints/samples_24j/2A+2Lo_0001_gvhmr_24j.npy',
    '2A+2Lo_0001_hybrik': 'data/smpl_joints/samples_24j/2A+2Lo_0001_hybrik_24j.npy',
    '8_front_9_cam2_hybrik': 'data/smpl_joints/samples_24j/8_front_9_cam2_hybrik_24j.npy',
}
VARIANTS = ['clean', 'noise', 'shape']
FINEFS_SEL = os.path.join(HERE, 'finefs_selection.json')


def load_finefs(clip, kind17=False):
    """讀 FineFS 一筆跳躍。回傳 Y-up 公尺的關節：
    kind17=False：資料集裡 HybrIK 偵測的 SMPL 24 點（整支影片的檔案切出該段）。
    kind17=True ：官方 17 點（Z-up 轉 Y-up，再做 normalize_h36m17 的前處理）。"""
    sel = {c['clip']: c for c in json.load(open(FINEFS_SEL))['clips']}[clip]
    s, e = sel['frame_start'], sel['frame_end']
    root = '/home/allen/datasets/FineFS'
    # 2026-11-24: FineFS/hybrik/ and FineFS/skeleton/ were deleted (已刪). skeleton.zip remains; hybrik has no copy.
    if not kind17:
        return np.load(f"{root}/hybrik/{sel['video']}/smpl24.npy")[s:e].astype(np.float32)
    from utils.axis_converter import convert_joints_z_to_y
    from methods import normalize_h36m17
    j = convert_joints_z_to_y(np.load(f"{root}/skeleton/{sel['video']}.npz")['reconstruction'][s:e])
    return normalize_h36m17(j)[0]


def load_amass_gt(npz_path, frames, variant):
    """回傳 (輸入關節, 標準答案 dict)。標準答案的手掌旋轉（22、23）歸零，因為 24 個關節點看不出來。"""
    import torch
    from common import ROOT, fk
    b = np.load(npz_path, allow_pickle=True)
    fps = float(b['mocap_framerate']) if 'mocap_framerate' in b else 20.0
    step = max(1, int(fps / 20))
    poses = b['poses'][::step, :72].astype(np.float32).copy()
    trans = b['trans'][::step].astype(np.float32)
    poses[:, 66:] = 0
    T = poses.shape[0]
    s = max(0, (T - frames) // 2)
    poses, trans = poses[s:s + frames], trans[s:s + frames]
    gender = str(b['gender']).replace("b'", '').replace("'", '').lower() if 'gender' in b else 'neutral'
    if gender not in ('male', 'female', 'neutral'):
        gender = 'neutral'
    betas = b['betas'][:10].astype(np.float32)
    seed = sum(map(ord, os.path.basename(npz_path)))  # 每個片段固定的亂數種子
    rng = np.random.RandomState(seed)
    if variant == 'shape':
        betas = rng.normal(0, 1.5, size=10).astype(np.float32)
    # AMASS 是 Z-up，本專案標準是 Y-up：整體朝向與位移繞 X 轉 -90 度
    from utils.axis_converter import convert_smpl_z_to_y
    d = convert_smpl_z_to_y({'poses': poses.astype(np.float64), 'trans': trans})
    gt = {'poses': d['poses'].astype(np.float32), 'betas': betas, 'trans': d['trans'].astype(np.float32),
          'gender': gender}
    with torch.no_grad():
        joints = fk(gt['poses'], gt['betas'], gt['trans'], gender).numpy()
    if variant == 'noise':
        joints = joints + rng.normal(0, 0.01, size=joints.shape).astype(np.float32)  # 每軸標準差 1 公分
    return joints.astype(np.float32), gt


def turns_from_joints(j):
    """由左右髖連線在水平面上的方向累計轉了幾圈（Y-up）。"""
    v = j[:, 1] - j[:, 2]
    yaw = np.unwrap(np.arctan2(v[:, 2], v[:, 0]))
    return float((yaw[-1] - yaw[0]) / (2 * np.pi)), yaw


def one_task(task):
    kind, clip, variant, method, src, frames = task
    tag = f'{kind}__{clip}__{variant}__{method}'
    out_json = os.path.join(RESULTS, tag + '.json')
    if os.path.exists(out_json):
        return tag, 'skip'
    import torch
    torch.set_num_threads(THREADS)
    from common import ROOT, fk, evaluate, accel
    from methods import METHODS
    os.chdir(ROOT)
    try:
        if kind == 'amass':
            joints, gt = load_amass_gt(src, frames, 'clean' if variant == 'h36m17' else variant)
            if variant == 'h36m17':     # 用標準答案的網格回歸出理想的 17 點當輸入，量「只有 17 點」本身損失多少
                from common import fk as _fk
                from methods import h36m_regressor
                with torch.no_grad():
                    v = _fk(gt['poses'], gt['betas'], gt['trans'], gt['gender'], verts=True)[1]
                    joints = torch.einsum('jv,tvc->tjc', h36m_regressor(), v).numpy()
        elif kind == 'finefs':
            joints, gt = load_finefs(clip, kind17=(variant == 'official17')), None
        else:
            joints, gt = np.load(src).astype(np.float32), None
        # 計時前先把模型與套件載好，時間只算擬合本身（EasyMocap 與手寫版本內部仍會各自載一次模型）
        from common import get_model
        import methods as M
        get_model('neutral')
        if method == 'hybrik_analytic':
            M._load_hybrik_lbs()
        if method == 'vposer_opt':
            M._load_vposer()
        t0 = time.time()
        pred = METHODS[method](joints)
        dt = time.time() - t0
        T = joints.shape[0]
        res = {'kind': kind, 'clip': clip, 'variant': variant, 'method': method, 'frames': T,
               'sec': dt, 'sec_per_1000_frames': dt / T * 1000}
        with torch.no_grad():
            pj = fk(pred['poses'], pred['betas'], pred['trans'], pred['gender']).numpy()
        pin = pj
        if joints.shape[1] == 17:
            from methods import regress_h36m17
            pin = regress_h36m17(pred)
        res['fit_to_input_mm'] = float(np.linalg.norm(pin - joints, axis=-1).mean() * 1000)
        res['fit_to_input_max_mm'] = float(np.linalg.norm(pin - joints, axis=-1).max() * 1000)
        if gt is not None:
            res.update(evaluate(pred, gt))
        elif joints.shape[1] == 17:
            res['accel_pred'] = accel(pin, 25.0)
            res['accel_input'] = accel(joints, 25.0)
        else:
            fps = 30.0
            res['accel_pred'] = accel(pj, fps)
            res['accel_input'] = accel(joints, fps)
            res['turns_input'] = turns_from_joints(joints)[0]
            res['turns_pred'] = turns_from_joints(pj)[0]
            # 逐幀的朝向差（度）：模型的髖連線方向 vs 輸入的髖連線方向
            yi, yp = turns_from_joints(joints)[1], turns_from_joints(pj)[1]
            d = np.rad2deg(np.abs(np.angle(np.exp(1j * (yi - yp)))))
            res['heading_err_mean_deg'] = float(d.mean())
            res['heading_err_max_deg'] = float(d.max())
        with open(os.path.join(RESULTS, tag + '.pkl'), 'wb') as f:
            pickle.dump({**{k: np.asarray(v) if k != 'gender' else v for k, v in pred.items()},
                         'mocap_framerate': {'skating': 30.0, 'finefs': 25.0}.get(kind, 20.0)}, f)
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=1)
        return tag, f"ok {dt:.0f}s fit={res['fit_to_input_mm']:.1f}mm"
    except Exception:
        with open(os.path.join(RESULTS, tag + '.err'), 'w') as f:
            f.write(traceback.format_exc())
        return tag, 'ERROR ' + traceback.format_exc().strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['amass', 'skating', 'finefs'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--methods', nargs='*')
    ap.add_argument('--clips', nargs='*')
    ap.add_argument('--variants', nargs='*', default=VARIANTS)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    from methods import METHODS
    methods = a.methods or list(METHODS)
    root = os.path.abspath(os.path.join(HERE, '..', '..'))
    tasks = []
    if a.kind == 'amass':
        for p in sorted(glob.glob(os.path.join(root, 'data/amass/samples_smpl_H_G/*.npz'))):
            clip = os.path.basename(p).replace('_poses.npz', '').replace('_sync', '')
            if a.clips and clip not in a.clips:
                continue
            for v in a.variants:
                for m in methods:
                    tasks.append(('amass', clip, v, m, p, a.frames))
    elif a.kind == 'finefs':
        m24 = ['easymocap_fixed', 'handmade', 'hybrik_analytic', 'vposer_opt', 'vposer_opt_wristfix']
        for c in json.load(open(FINEFS_SEL))['clips']:
            for m in (a.methods or m24):
                if m != 'vposer_opt_h36m17':
                    tasks.append(('finefs', c['clip'], 'hybrik24', m, None, 0))
            if not a.methods or 'vposer_opt_h36m17' in a.methods:
                tasks.append(('finefs', c['clip'], 'official17', 'vposer_opt_h36m17', None, 0))
    else:
        for clip, rel in SKATING.items():
            for m in methods:
                tasks.append(('skating', clip, 'input', m, os.path.join(root, rel), 0))
    # 慢的方法先排
    order = {'handmade': 0, 'easymocap_orig': 1, 'easymocap_fixed': 1, 'easymocap_fixed_order_only': 1}
    tasks.sort(key=lambda t: order.get(t[3], 2))
    print(f'共 {len(tasks)} 個組合，{a.workers} 個行程，每個 {THREADS} 條執行緒', flush=True)
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(a.workers, maxtasksperchild=4) as pool:
        for i, (tag, msg) in enumerate(pool.imap_unordered(one_task, tasks), 1):
            print(f'[{i}/{len(tasks)}] {time.time() - t0:6.0f}s {tag}: {msg}', flush=True)
    print('全部完成', flush=True)


if __name__ == '__main__':
    main()
