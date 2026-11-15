"""把 results/*.json 彙整成 Markdown 表格，印到標準輸出。"""
import os
import sys
import glob
import json
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('J2S_RESULTS', os.path.join(HERE, 'results'))
NAMES = {
    'easymocap_orig': '現有版本原樣',
    'easymocap_fixed_order_only': 'EasyMocap，只修關節對應',
    'easymocap_fixed': 'EasyMocap，修關節對應與位移換算',
    'handmade': '手寫擬合程式（原樣）',
    'hybrik_analytic': 'HybrIK 解析式',
    'vposer_opt': 'VPoser 加最佳化',
    'vposer_opt_wristfix': 'VPoser 加最佳化，手腕修正版',
}
VARIANT = {'clean': '原始關節', 'noise': '加 1 公分雜訊', 'shape': '換體型'}
rows = [json.load(open(f)) for f in glob.glob(os.path.join(RES, '*.json'))]


def table(kind, variant, cols, header):
    g = defaultdict(list)
    for r in rows:
        if r['kind'] == kind and r['variant'] == variant:
            g[r['method']].append(r)
    print('| 版本 | ' + ' | '.join(header) + ' | 片段數 |')
    print('| :--- | ' + ' | '.join(['---:'] * len(header)) + ' | ---: |')
    for m in NAMES:
        if m in g:
            vals = [np.mean([r[c] for r in g[m]]) for c in cols]
            print(f'| {NAMES[m]} | ' + ' | '.join(f'{v:.1f}' for v in vals) + f' | {len(g[m])} |')
    print()


for v in VARIANT:
    print(f'#### {VARIANT[v]}\n')
    table('amass', v,
          ['joint_mm', 'joint_max_mm', 'rot_main_deg', 'limb_orient_main_deg', 'vertex_mm', 'sec_per_1000_frames'],
          ['關節誤差 平均（公釐）', '關節誤差 最大（公釐）', '關節旋轉誤差（度）', '肢體朝向誤差（度）', '網格頂點誤差（公釐）', '每 1000 幀秒數'])

print('#### 各片段的關節誤差（公釐，原始關節輸入）\n')
clips = sorted({r['clip'] for r in rows if r['kind'] == 'amass'})
ms = [m for m in NAMES if any(r['method'] == m for r in rows)]
print('| 片段 | ' + ' | '.join(NAMES[m] for m in ms) + ' |')
print('| :--- | ' + ' | '.join(['---:'] * len(ms)) + ' |')
for c in clips:
    cells = []
    for m in ms:
        r = [r for r in rows if r['kind'] == 'amass' and r['clip'] == c and r['variant'] == 'clean' and r['method'] == m]
        cells.append(f"{r[0]['joint_mm']:.1f}" if r else '-')
    print(f'| {c} | ' + ' | '.join(cells) + ' |')
print()

print('#### 加速度（抖動），AMASS 原始關節輸入：擬合結果 ÷ 標準答案\n')
print('| 版本 | 比值（1 表示和標準答案一樣平順） |\n| :--- | ---: |')
for m in ms:
    r = [r for r in rows if r['kind'] == 'amass' and r['variant'] == 'clean' and r['method'] == m]
    if r:
        print(f"| {NAMES[m]} | {np.mean([x['accel_pred'] / x['accel_gt'] for x in r]):.2f} |")
print()

print('#### 滑冰\n')
print('| 片段（偵測器） | 版本 | 貼合輸入 平均（公釐） | 最大（公釐） | 抖動比（擬合÷輸入） | 輸入圈數 | 擬合圈數 | 朝向差 平均（度） | 朝向差 最大（度） |')
print('| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for r in sorted([r for r in rows if r['kind'] == 'skating'], key=lambda r: (r['clip'], list(NAMES).index(r['method']))):
    print(f"| {r['clip']} | {NAMES[r['method']]} | {r['fit_to_input_mm']:.1f} | {r['fit_to_input_max_mm']:.1f} | "
          f"{r['accel_pred'] / r['accel_input']:.2f} | {r['turns_input']:.2f} | {r['turns_pred']:.2f} | "
          f"{r['heading_err_mean_deg']:.1f} | {r['heading_err_max_deg']:.1f} |")
