"""把同一段輸入、各版本擬合出來的 SMPL 網格並排畫成一支影片。純 CPU（OpenCV 畫三角形），不經過顯示卡。

專案原本的 utils/rendering/mesh_renderer.py 用 EGL，會用到顯示卡；這台機器沒有 OSMesa，
所以這裡改用「依深度排序後逐面填色」的簡易畫法。畫質比不上原本的渲染，但足夠看出手腳有沒有翻轉。

用法： python experiments/joints_to_smpl/render_compare.py skating   （或 amass --clips jumping_jacks）
每支影片：上排正面視角、中排側面視角、下排右手特寫；每欄一個版本；紅點是輸入的關節點。
"""
import os
import sys
import glob
import pickle
import argparse
import subprocess
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, fk, get_model, BONES  # noqa: E402
import run_eval  # noqa: E402

PW, PH = 340, 440          # 每格大小
VIEW_H = 2.3               # 每格可見高度（公尺）
LABELS = {
    'easymocap_fixed': '1 EasyMocap fixed',
    'handmade': '2 handmade solver',
    'hybrik_analytic': '3 HybrIK analytic',
    'vposer_opt': '4 VPoser+optim',
    'vposer_opt_wristfix': '5 = 4 + wrist fix',
    'relaxed_hand': '6 = 5 + relaxed hand',
}
ZOOM_H = 0.55              # 手部特寫每格可見高度（公尺）


def project(P, center, view, view_h=VIEW_H):
    """正交投影。view='front' 沿 -Z 看（畫面右方是 +X）；'side' 沿 -X 看（畫面右方是 -Z）。回傳像素座標與深度。"""
    Q = P - center
    s = PH / view_h
    if view == 'front':
        x, depth = Q[..., 0], Q[..., 2]
    else:
        x, depth = -Q[..., 2], Q[..., 0]
    u = PW / 2 + x * s
    v = PH / 2 - Q[..., 1] * s
    return np.stack([u, v], -1), depth


def draw_mesh(img, V, F, center, view, color, view_h=VIEW_H):
    uv, depth = project(V, center, view, view_h)
    tri = uv[F]                                   # (F,3,2)
    d = depth[F].mean(1)
    e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    n = np.cross(e1, e2)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    view_dir = np.array([0, 0, 1.0]) if view == 'front' else np.array([1.0, 0, 0])   # 朝向觀察者
    facing = n @ view_dir
    keep = facing > 0
    light = np.array([0.3, 0.5, 0.8]) if view == 'front' else np.array([0.8, 0.5, -0.3])
    light = light / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    order = np.argsort(d)                          # 遠的先畫
    tri_i = np.round(tri * 8).astype(np.int32)
    base = np.array(color, dtype=np.float64)
    for i in order:
        if keep[i]:
            c = base * shade[i]
            cv2.fillConvexPoly(img, tri_i[i], (int(c[0]), int(c[1]), int(c[2])), lineType=cv2.LINE_AA, shift=3)


def draw_skeleton(img, J, center, view, color=(60, 60, 60), dots_only=False, view_h=VIEW_H):
    uv, _ = project(J, center, view, view_h)
    if not dots_only:
        from methods import H36M_BONES
        for p, c in (BONES if J.shape[0] == 24 else H36M_BONES):
            cv2.line(img, tuple(np.round(uv[p]).astype(int)), tuple(np.round(uv[c]).astype(int)), color, 2, cv2.LINE_AA)
    for j in range(J.shape[0]):
        cv2.circle(img, tuple(np.round(uv[j]).astype(int)), 3, (0, 0, 230), -1, cv2.LINE_AA)


def render_clip(name, joints, preds, out_path, fps):
    """joints: 輸入 (T,24,3)；preds: {方法名: dict}"""
    T = joints.shape[0]
    faces = get_model('neutral').faces.astype(np.int64)
    verts = {}
    with torch.no_grad():
        for m, p in preds.items():
            verts[m] = fk(p['poses'], p['betas'], p['trans'], p['gender'], verts=True)[1].numpy()
    if 'vposer_opt_wristfix' in preds:
        from methods import to_smplh_relaxed
        ph, vh = to_smplh_relaxed(preds['vposer_opt_wristfix'], joints)
        verts['relaxed_hand'] = vh
        preds = dict(preds, relaxed_hand=ph)
        with open(out_path.replace('.mp4', '__wristfix_relaxed_hand_smplh.pkl'), 'wb') as f:
            pickle.dump(dict(ph, mocap_framerate=fps), f)
    cols = ['input'] + list(preds)
    W, H = PW * len(cols), PH * 3
    tmp = out_path.replace('.mp4', '_tmp.mp4')
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    # 鏡頭跟著輸入的骨盆走（平滑過），高度固定在整段的中間
    k = max(1, int(fps // 3))
    pel = joints[:, 0].copy()
    pad = np.pad(pel, ((k, k), (0, 0)), mode='edge')
    pel = np.stack([np.convolve(pad[:, a], np.ones(2 * k + 1) / (2 * k + 1), mode='valid') for a in range(3)], 1)
    for t in range(T):
        frame = np.full((H, W, 3), 255, np.uint8)
        center = pel[t].copy()
        center[1] = pel[t, 1] + 0.1
        for r, (view, tag) in enumerate([('front', 'front'), ('side', 'side'), ('front', 'right hand')]):
            # 第三排是右手特寫：鏡頭中心放在輸入的右手腕
            ctr, vh_ = (center, VIEW_H) if r < 2 else (joints[t, 21], ZOOM_H)
            for c, m in enumerate(cols):
                panel = np.full((PH, PW, 3), 255, np.uint8)
                if m == 'input':
                    draw_skeleton(panel, joints[t], ctr, view, view_h=vh_)
                else:
                    draw_mesh(panel, verts[m][t], faces, ctr, view, (220, 170, 110), view_h=vh_)
                    draw_skeleton(panel, joints[t], ctr, view, dots_only=True, view_h=vh_)
                label = 'input joints' if m == 'input' else LABELS.get(m, m)
                cv2.putText(panel, f'{label} [{tag}]', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(panel, (0, 0), (PW - 1, PH - 1), (200, 200, 200), 1)
                frame[r * PH:(r + 1) * PH, c * PW:(c + 1) * PW] = panel
        cv2.putText(frame, f'{name}  frame {t}/{T - 1}', (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '23', out_path], check=True)
    os.remove(tmp)
    print('已寫出', out_path, flush=True)


def render_finefs17(clip, j24, j17, pred17, pred24fix, out_path, fps=25.0):
    """官方 17 點路徑的精簡影片，五欄：
    官方 17 點輸入、由 17 點擬合的 SMPL、再換放鬆手型、（對照）HybrIK 24 點輸入、（對照）由 24 點擬合的第 6 欄。
    兩種骨架的座標系不同（官方 17 點以骨盆為原點且重新縮放過），所以每格各自以自己的骨盆為中心，
    左三欄和右兩欄的朝向不保證一致。"""
    from methods import to_smplh_relaxed, regress_h36m17
    faces = get_model('neutral').faces.astype(np.int64)
    with torch.no_grad():
        v17 = fk(pred17['poses'], pred17['betas'], pred17['trans'], pred17['gender'], verts=True)[1].numpy()
        p17j = fk(pred17['poses'], pred17['betas'], pred17['trans'], pred17['gender']).numpy()
    # 放鬆手型需要 24 點的骨長來估 SMPL-H 體型，這裡用擬合結果自己的 24 關節
    ph17, vh17 = to_smplh_relaxed(pred17, p17j)
    with open(out_path.replace('.mp4', '__relaxed_hand_smplh.pkl'), 'wb') as f:
        pickle.dump(dict(ph17, mocap_framerate=fps), f)
    _, vh24 = to_smplh_relaxed(pred24fix, j24)
    cols = [('official 17 joints', 'skel', j17, j17), ('A fit to 17 (VPoser)', 'mesh', v17, j17),
            ('B = A + relaxed hand', 'mesh', vh17, j17), ('HybrIK 24 joints', 'skel', j24, j24),
            ('6 from 24 joints', 'mesh', vh24, j24)]
    T = j17.shape[0]
    W, H = PW * len(cols), PH * 2
    tmp = out_path.replace('.mp4', '_tmp.mp4')
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for t in range(T):
        frame = np.full((H, W, 3), 255, np.uint8)
        for r, view in enumerate(['front', 'side']):
            for c, (label, kind, data, dots) in enumerate(cols):
                panel = np.full((PH, PW, 3), 255, np.uint8)
                ctr = dots[t, 0].copy()
                ctr[1] += 0.1
                if kind == 'skel':
                    draw_skeleton(panel, data[t], ctr, view)
                else:
                    draw_mesh(panel, data[t], faces, ctr, view, (220, 170, 110))
                    draw_skeleton(panel, dots[t], ctr, view, dots_only=True)
                cv2.putText(panel, f'{label} [{view}]', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(panel, (0, 0), (PW - 1, PH - 1), (200, 200, 200), 1)
                frame[r * PH:(r + 1) * PH, c * PW:(c + 1) * PW] = panel
        cv2.putText(frame, f'{clip} (FineFS official 17 joints)  frame {t}/{T - 1}', (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '23', out_path], check=True)
    os.remove(tmp)
    print('已寫出', out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['skating', 'amass', 'finefs'])
    ap.add_argument('--clips', nargs='*')
    ap.add_argument('--variant', default='clean')
    ap.add_argument('--fps', type=float)
    a = ap.parse_args()
    res = run_eval.RESULTS
    out_dir = os.path.join(HERE, 'videos')
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(ROOT)
    if a.kind == 'finefs':
        import json
        for c in json.load(open(run_eval.FINEFS_SEL))['clips']:
            clip = c['clip']
            if a.clips and clip not in a.clips:
                continue
            j24 = run_eval.load_finefs(clip)
            L = lambda v, m: pickle.load(open(os.path.join(res, f'finefs__{clip}__{v}__{m}.pkl'), 'rb'))
            preds = {m: L('hybrik24', m) for m in LABELS if m != 'relaxed_hand'}
            render_clip(f"FineFS {clip} (HybrIK 24 joints)", j24, preds, os.path.join(out_dir, f'finefs__{clip}__hybrik24.mp4'), 25.0)
            render_finefs17(clip, j24, run_eval.load_finefs(clip, kind17=True), L('official17', 'vposer_opt_h36m17'),
                            preds['vposer_opt_wristfix'], os.path.join(out_dir, f'finefs__{clip}__official17.mp4'))
        return
    if a.kind == 'skating':
        items = [(c, 'input', np.load(os.path.join(ROOT, p)).astype(np.float32)) for c, p in run_eval.SKATING.items()]
    else:
        items = []
        for p in sorted(glob.glob(os.path.join(ROOT, 'data/amass/samples_smpl_H_G/*.npz'))):
            c = os.path.basename(p).replace('_poses.npz', '').replace('_sync', '')
            items.append((c, a.variant, run_eval.load_amass_gt(p, 150, a.variant)[0]))
    for clip, variant, joints in items:
        if a.clips and clip not in a.clips:
            continue
        preds = {}
        for m in [k for k in LABELS if k != 'relaxed_hand']:
            f = os.path.join(res, f'{a.kind}__{clip}__{variant}__{m}.pkl')
            if os.path.exists(f):
                preds[m] = pickle.load(open(f, 'rb'))
        if not preds:
            print('沒有結果，跳過', clip)
            continue
        # 2A+2Lo 原片 25 fps；8_front_9 原片約 60 fps，這裡用 30 fps 寫出，等於半速播放
        fps = a.fps or ({'2A+2Lo_0001_gvhmr': 25.0, '2A+2Lo_0001_hybrik': 25.0}.get(clip, 30.0) if a.kind == 'skating' else 20.0)
        render_clip(f'{clip} ({variant})', joints, preds, os.path.join(out_dir, f'{a.kind}__{clip}__{variant}.mp4'), fps)


if __name__ == '__main__':
    main()
