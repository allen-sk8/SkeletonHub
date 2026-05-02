import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# 強制使用 Agg 後端以支援無介面伺服器
import matplotlib
matplotlib.use('Agg')

# --- 骨架層級結構與顏色定義 ---

# 🌟 預設改為 Standard SMPL 24 joints (符合使用者需求)
SMPL_24_CHAIN = [
    [0, 2, 5, 8, 11], # Right leg
    [0, 1, 4, 7, 10], # Left leg
    [0, 3, 6, 9, 12, 15], # Spine & Head
    [9, 14, 17, 19, 21, 23], # Right arm (includes hand)
    [9, 13, 16, 18, 20, 22]  # Left arm (includes hand)
]
SMPL_24_COLORS = ['red', 'blue', 'black', 'red', 'blue']

# HumanML3D 22 joints (作為備選定義)
H3D_22_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
H3D_22_COLORS = ['red', 'blue', 'black', 'red', 'blue']

def plot_xz_plane(ax, min_x, max_x, min_y, min_z, max_z):
    """繪製灰色的地平面 (XZ Plane)"""
    verts = [
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, min_y, min_z]
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.2))
    ax.add_collection3d(xz_plane)

def render_motion(joints, save_path, title="Motion Visualization", fps=20, radius=3, kinematic_chain=None, colors=None):
    """
    標準化 3D 動作渲染核心
    joints: (T, J, 3)
    """
    # 🌟 若外部沒給定義，預設使用 SMPL 24j 配置
    if kinematic_chain is None: kinematic_chain = SMPL_24_CHAIN
    if colors is None: colors = SMPL_24_COLORS

    data = joints.copy()
    frame_number = data.shape[0]

    # 初始化畫布
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # 軌跡紀錄 (Root trajectory)
    trajectory = data[:, 0, [0, 2]] # X, Z

    def update(index):
        while ax.lines:
            ax.lines[0].remove()
        while ax.collections:
            ax.collections[0].remove()
        while ax.texts:
            ax.texts[0].remove()
        
        # 強制標準視角
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        
        # 固定尺度
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([0, radius/2])
        ax.set_zlim3d([-radius, radius])
        ax.set_box_aspect((2*radius, radius/2, 2*radius))
        
        ax.set_xlabel('X (Lateral)')
        ax.set_ylabel('Y (Height)')
        ax.set_zlabel('Z (Forward)')
        ax.set_title(f"{title}\nFrame {index}/{frame_number}")
        
        ax.text2D(0.05, 0.95, "Right (Red)", transform=ax.transAxes, color='red', fontsize=12, fontweight='bold')
        ax.text2D(0.05, 0.90, "Left (Blue)", transform=ax.transAxes, color='blue', fontsize=12, fontweight='bold')
        ax.text2D(0.05, 0.85, "Center (Black)", transform=ax.transAxes, color='black', fontsize=12, fontweight='bold')

        plot_xz_plane(ax, -radius, radius, 0, -radius, radius)

        if index > 1:
            ax.plot3D(trajectory[:index, 0], 
                      np.zeros_like(trajectory[:index, 0]), 
                      trajectory[:index, 1], 
                      linewidth=1.0, color='blue', alpha=0.3)

        # 繪製骨架
        curr_data = data[index]
        for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            linewidth = 4.0 if i < 3 else 2.0
            ax.plot3D(curr_data[chain, 0], curr_data[chain, 1], curr_data[chain, 2], linewidth=linewidth, color=color)

        '''[DEBUG] 繪製關節點與編號
        # 一般關節點用灰色小點
        ax.scatter(curr_data[:, 0], curr_data[:, 1], curr_data[:, 2], s=10, c='gray', alpha=0.3)
        
        # 特別標註手部關節 (20, 21, 22, 23)
        hand_indices = [idx for idx in [20, 21, 22, 23] if idx < len(curr_data)]
        if hand_indices:
            ax.scatter(curr_data[hand_indices, 0], curr_data[hand_indices, 1], curr_data[hand_indices, 2], 
                       s=40, c='gold', edgecolors='black', label='Hand Joints')
            for idx in hand_indices:
                ax.text(curr_data[idx, 0], curr_data[idx, 1], curr_data[idx, 2], f" {idx}", color='darkorange', fontsize=8)
        '''

        ax.grid(True)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    
    print(f"🎬 正在渲染影片至: {save_path} ...")
    ani.save(save_path, fps=fps, writer='ffmpeg')
    plt.close()
