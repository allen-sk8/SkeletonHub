import os
import numpy as np
import trimesh
import pyrender
import cv2
import tqdm

os.environ['PYOPENGL_PLATFORM'] = 'egl'

class MeshRenderer:
    def __init__(self, width=1024, height=1024, background_color=[1.0, 1.0, 1.0, 1.0]):
        self.width = width
        self.height = height
        self.bg_color = background_color # 改為白色背景

    def _create_ground_plane(self, center_x=0, center_z=0, floor_y=0.0, radius=10):
        verts = np.array([
            [center_x - radius, floor_y, center_z - radius],
            [center_x + radius, floor_y, center_z - radius],
            [center_x + radius, floor_y, center_z + radius],
            [center_x - radius, floor_y, center_z + radius]
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        ground = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=1.0,
            baseColorFactor=[0.9, 0.9, 0.9, 1.0], # 不透明淺灰色地面
            doubleSided=True
        )
        return pyrender.Mesh.from_trimesh(ground, material=material)

    def _create_axes(self, origin=[0, 0, 0], length=0.2):
        transform = np.eye(4)
        transform[:3, 3] = origin
        axis_mesh = trimesh.creation.axis(origin_size=0.01, axis_radius=0.005, axis_length=length, transform=transform)
        return pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)

    def render_motion(self, vertices, faces, save_path, fps=20, title="SMPL Mesh", axis_length=0.2):
        T = vertices.shape[0]
        
        # 🌟 計算人物中心點，避免相機拍空
        center = np.mean(vertices[0], axis=0)
        print(f"📍 人物初始中心點: {center}")
        
        # 🌟 找出最低點，把地板放在腳底
        floor_y = np.min(vertices[:, :, 1])
        print(f"📍 地板高度設定為: {floor_y}")

        scene = pyrender.Scene(bg_color=self.bg_color, ambient_light=[0.5, 0.5, 0.5])
        
        # 加入在地心位置的地平面與座標軸
        scene.add(self._create_ground_plane(center_x=center[0], center_z=center[2], floor_y=floor_y))
        scene.add(self._create_axes(origin=[0, 0, 0], length=axis_length))
        
        # 🌟 動態設定相機位置，指向人物中心
        # camera_pos: [左右偏移, 高度, 前後距離]
        # camera_target: 相機注視的點 (y + 1.0 約為胸部高度)
        camera_pos = np.array([center[0], center[1] + 1.2, center[2] + 4.0])
        camera_target = np.array([center[0], center[1] + 0, center[2]])
        
        # 計算相機矩陣 (Look-at)
        forward = camera_pos - camera_target
        forward /= np.linalg.norm(forward)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = forward
        camera_pose[:3, 3] = camera_pos
        
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
        scene.add(camera, pose=camera_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene.add(light, pose=camera_pose)
        
        r = pyrender.OffscreenRenderer(self.width, self.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(save_path, fourcc, fps, (self.width, self.height))
        
        print(f"🎬 正在渲染 SMPL Mesh 影片...")
        for t in tqdm.tqdm(range(T)):
            mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2, roughnessFactor=0.8,
                baseColorFactor=[0.2, 0.5, 0.8, 1.0] # 深天藍色
            )
            mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material)
            node = scene.add(mesh_node)
            
            # 渲染並寫入
            color, _ = r.render(scene)
            color = color.copy()
            
            # 加入 XYZ 標示 (在左上角)
            cv2.putText(color, "X (Red)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(color, "Y (Green)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color, "Z (Blue)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            vw.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            
            scene.remove_node(node)
            
        vw.release()
        r.delete()

        # H.264 重新編碼
        print(f"🔄 正在優化影片編碼 (H.264)...")
        temp_path = save_path.replace(".mp4", "_temp.mp4")
        os.rename(save_path, temp_path)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", temp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", save_path]
        try:
            import subprocess
            subprocess.run(cmd, check=True)
            os.remove(temp_path)
        except:
            os.rename(temp_path, save_path)
