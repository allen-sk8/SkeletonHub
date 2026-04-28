import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys
# 修正路徑：從 data/humanml3d/scripts 回到專案根目錄需要 3 層上移
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)
from converters.humanml3d_263d_to_joints import convert_263d_to_joints

# HumanML3D 22 Joints Names
JOINT_NAMES = [
    'Root', 'L_Hip', 'R_Hip', 'Spine_Low', 'L_Knee', 'R_Knee', 'Spine_Mid',
    'L_Ankle', 'R_Ankle', 'Spine_Mid', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
]

KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def profile_from_real_data(npy_path, save_path):
    """
    優化排版後的骨架定義圖
    """
    print(f"📖 正在載入樣本數據: {npy_path}")
    joints = convert_263d_to_joints(npy_path)
    frame_idx = 0
    data = joints[frame_idx]
    data = data - data[0] # 以 Root 為原點
    # 全身上移約 0.2m
    data[:, 1] += 0.5
    
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # 1. 繪製骨架線條
    for chain in KINEMATIC_CHAIN:
        ax.plot(data[chain, 0], data[chain, 1], 'o-', linewidth=3, markersize=6, color='#34495e', alpha=0.7)
        
        # 標註骨骼長度 (放在骨骼旁邊，縮小字級)
        for i in range(len(chain)-1):
            p1 = data[chain[i]]
            p2 = data[chain[i+1]]
            length = np.linalg.norm(p2 - p1)
            mid = (p1 + p2) / 2
            # 計算垂直於骨骼的偏移向量以避免重疊
            vec = p2 - p1
            perp = np.array([-vec[1], vec[0]])
            if np.linalg.norm(perp) > 0:
                perp = perp / np.linalg.norm(perp) * 0.04
            
            ax.text(mid[0] + perp[0], mid[1] + perp[1], f"{length:.2f}m", 
                    color='#27ae60', fontsize=8, fontweight='bold', alpha=0.8)

    # 2. 標註關節 ID 與名稱 (自動位移邏輯)
    for i in range(len(data)):
        x, y = data[i, 0], data[i, 1]
        
        # 根據關節位置決定文字偏置方向
        ha = 'left'
        va = 'bottom'
        ox, oy = 0.02, 0.02
        
        if x < -0.01: # 左側節點 (人體的右側)
            ha = 'right'
            ox = -0.02
        if y < 0.2: # 下半身
            va = 'top'
            oy = -0.02
        if i in [15, 12]: # 頭頸部向上偏
            va = 'bottom'
            oy = 0.05
            
        # 繪製 ID (大字) 與 名稱 (小字)
        ax.text(x + ox, y + oy, f"#{i}", color='#2980b9', fontsize=12, fontweight='black', ha=ha, va=va)
        ax.text(x + ox, y + oy - (0.04 if va=='top' else -0.04), f"{JOINT_NAMES[i]}", 
                color='#7f8c8d', fontsize=8, ha=ha, va=va, alpha=0.9)

    # 3. 標註 L / R (放在外側)
    ax.text(-0.7, 1.0, "RIGHT SIDE\n(Red)", color='#c0392b', fontsize=16, fontweight='bold', ha='center', alpha=0.2)
    ax.text(0.7, 1.0, "LEFT SIDE\n(Blue)", color='#2980b9', fontsize=16, fontweight='bold', ha='center', alpha=0.2)

    # 4. 輔助資訊與座標軸
    plt.title("HumanML3D 22-Joint Definition Profile", fontsize=18, pad=30, fontweight='bold')
    
    # 座標軸示意 (左下角)
    ax.annotate('', xy=(-0.7, -0.4), xytext=(-0.85, -0.4), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(-0.88, -0.4, 'X', va='center', fontweight='bold')
    ax.annotate('', xy=(-0.85, -0.25), xytext=(-0.85, -0.4), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(-0.85, -0.2, 'Y', ha='center', fontweight='bold')
    ax.text(-0.85, -0.48, 'Z ➔ into page', fontsize=9, fontstyle='italic')

    # 設定範圍與美化
    ax.set_aspect('equal')
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.8, 2.1) # 增加下方空間避免腳部被切到
    ax.axis('off')
    
    # 底部規範說明欄 (移至右下角避免擋到腳部標籤)
    info_box = (
        " TECHNICAL SPECIFICATIONS\n"
        " -----------------------\n"
        " Dataset  : HumanML3D\n"
        " Unit     : Meters (m)\n"
        " Up       : Y+ Axis\n"
        " Forward  : Z+ Axis\n"
        " Mapping  : SMPL 0-21 Layout"
    )
    plt.figtext(0.6, 0.1, info_box, fontsize=10, family='monospace', 
                bbox=dict(facecolor='white', edgecolor='#bdc3c7', boxstyle='round,pad=1'))

    print(f"🖼️ 正在儲存優化後的規格圖至: {save_path}")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 取得當前腳本目錄並推算相對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.dirname(script_dir)
    
    npy_sample = os.path.join(dataset_dir, "samples/012314.npy")
    save_path = os.path.join(dataset_dir, "skeleton_definition.jpg")
    
    if os.path.exists(npy_sample):
        profile_from_real_data(npy_sample, save_path)
    else:
        print(f"❌ 找不到樣本檔案: {npy_sample}")
