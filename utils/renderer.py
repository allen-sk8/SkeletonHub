import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# 強制使用 Agg 後端以支援無介面伺服器
import matplotlib
matplotlib.use('Agg')

# 骨架層級結構與顏色定義 (HumanML3D 22 joints)
KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
COLORS = ['red', 'blue', 'black', 'red', 'blue', 'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkred', 'darkred','darkred','darkred','darkred']

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

def render_motion(joints, save_path, title="Motion Visualization", fps=20, radius=3):
    """
    標準化 3D 動作渲染核心
    joints: (T, J, 3)
    """
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
        
        # 🟢 強制標準視角
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        
        # 🟢 固定尺度一視同仁 (使用傳入的 radius 作為絕對邊界)
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([0, radius/2])
        ax.set_zlim3d([-radius, radius])
        
        # 🟢 修正比例：確保畫面呈現扁平狀但骨架不變形 (X:Z:Y = 2*rad : 2*rad : rad/2)
        ax.set_box_aspect((2*radius, radius/2, 2*radius))
        
        # 標示軸向與刻度
        ax.set_xlabel('X (Lateral)')
        ax.set_ylabel('Y (Height)')
        ax.set_zlabel('Z (Forward)')
        ax.set_title(f"{title}\nFrame {index}/{frame_number}")
        
        # 🟢 加入顏色標註 (Color Legend)
        ax.text2D(0.05, 0.95, "Right (Red)", transform=ax.transAxes, color='red', fontsize=12, fontweight='bold')
        ax.text2D(0.05, 0.90, "Left (Blue)", transform=ax.transAxes, color='blue', fontsize=12, fontweight='bold')
        ax.text2D(0.05, 0.85, "Center (Black)", transform=ax.transAxes, color='black', fontsize=12, fontweight='bold')

        # 繪製固定地平面
        plot_xz_plane(ax, -radius, radius, 0, -radius, radius)

        # 繪製根節點路徑 (世界座標)
        if index > 1:
            ax.plot3D(trajectory[:index, 0], 
                      np.zeros_like(trajectory[:index, 0]), 
                      trajectory[:index, 1], 
                      linewidth=1.0, color='blue', alpha=0.3)

        # 繪製骨架 (直接使用世界座標)
        curr_data = data[index]
        for i, (chain, color) in enumerate(zip(KINEMATIC_CHAIN, COLORS)):
            linewidth = 4.0 if i < 3 else 2.0
            ax.plot3D(curr_data[chain, 0], curr_data[chain, 1], curr_data[chain, 2], linewidth=linewidth, color=color)

        ax.grid(True) # 開啟網格以便判斷尺度
        # plt.axis('on') # 確保軸與刻度可見 (預設即為 on)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    
    print(f"🎬 正在渲染影片至: {save_path} ...")
    ani.save(save_path, fps=fps, writer='ffmpeg')
    plt.close()
