# SkeletonHub 施工紀錄與參考日誌 (PROGRESS)

本文件紀錄專案的開發進度、各項轉換器腳本的支援狀態、核心基礎設施進程以及對外部工具的參考紀錄。

---

## 1. 施工進度表 (Format Converters)

目前專案已完整實作 17 個核心格式與骨架轉換器，全部均支援**多級後綴串接自動命名**與 **`--vis` 一鍵自動視覺化渲染**：

| # | 來源格式 | 目標格式 | 轉換腳本 | 進度狀態 | 負責人 | 核心技術與備註 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **AMASS (.npz)** | SMPL-H (.pkl) | `amass_to_smplh.py` | ✅ 已完成 | Antigravity | 156D 保留完整手部指節與全域 Y-up 轉換 |
| 2 | **AMASS (.npz)** | SMPL (.pkl) | `amass_to_smpl.py` | ✅ 已完成 | Antigravity | 72D 基礎人體、10D betas、Y-up 轉換 |
| 3 | **SMPL-H (.pkl)** | SMPL (24j) | `smplh_to_smpl_24j.py` | ✅ 已完成 | Antigravity | 解決指節導致的左右手 22/37 索引偏移 |
| 4 | **SMPL-H (.pkl)** | SMPL-H (52j) | `smplh_to_smplh_52j.py` | ✅ 已完成 | Antigravity | 解算輸出包含 15 節雙手關節的 3D 座標 |
| 5 | **SMPL-H (.pkl)** | HumanML3D (22j) | `smplh_to_humanml3d_22j.py`| ✅ 已完成 | Antigravity | 用於 Text-to-Motion 訓練集對準與生成 |
| 6 | **SMPL-H (.pkl)** | BODY25 (.npy) | `smplh_to_body_25j.py` | ✅ 已完成 | Antigravity | 透過 EasyMocap 回歸矩陣直接由頂點回歸 |
| 7 | **SMPL (.pkl)** | HumanML3D (22j) | `smpl_to_humanml3d_22j.py` | ✅ 已完成 | Allen / Anti | 標準 24 關節前向動力學 FK 裁切為 22 關節 |
| 8 | **SMPL (.pkl)** | BODY25 (.npy) | `smpl_to_body_25j.py` | ✅ 已完成 | Antigravity | 標準 SMPL 頂點回歸 BODY25 關節，支持 Y-up |
| 9 | **SMPL (.pkl)** | SMPL (24j) | `smpl_to_smpl_24j.py` | ✅ 已完成 | Allen | 標準 24 關節前向動力學 FK (body only) |
| 10 | **Joints (24j)** | SMPL (.pkl) | `joints_24j_to_smpl.py` | ✅ 已完成 | Antigravity | 整合 EasyMocap 優化擬合 (PyTorch L-BFGS) |
| 11 | **Joints (24j)** | SMPL (.pkl) | `joints_24j_to_smpl_smplx_handmade.py` | ✅ 已完成 | Antigravity | 自研 PyTorch 手工解剖極限約束 IK Solver |
| 12 | **Joints (52j)** | SMPL-H (.pkl) | `joints_52j_to_smplh_easymocap.py` | ✅ 已完成 | Antigravity | 含雙手手指運動捕捉的高精度優化擬合 (EasyMocap) |
| 13 | **Joints (52j)** | SMPL-H (.pkl) | `joints_52j_to_smplh.py` | ✅ 已完成 | Antigravity | 利用 SMPLHandler 與 L2 Loss 的快速擬合器 |
| 14 | **Joints (52j)** | SMPL-H (.pkl) | `joints_52j_to_smplh_smplifyx.py` | ✅ 已完成| Antigravity | 整合 SMPLify-X 框架與 VPoser 先驗約束 |
| 15 | **HybrIK (.pk)** | Joints (24j) | `HybrIK_to_joints_24j.py` | ✅ 已完成 | Antigravity | 還原 2.2m 物理尺度，Y-down 轉全域 Y-up |
| 16 | **HumanML3D (22j)**| HumanML3D (263D)| `humanml3d_22j_to_humanml3d_263d.py` | ✅ 已完成 | Antigravity | 計算 RIC/Rotation 6D/Velocity/Foot Contacts |
| 17 | **HumanML3D (263D)**| HumanML3D (22j)| `humanml3d_263d_to_humanml3d_22j.py` | ✅ 已完成| Antigravity | 累積 Root 速度、重塑局部偏置反算 3D 座標 |
| 18 | **COCO-Wholebody (133j)** | BODY25 (.npy) | `joints_133j_to_body_25j.py` | ✅ 已完成 | Antigravity | 透過對應與插值 Neck/MidHip，支援影片一鍵自動視覺化 |

---

## 2. 核心基礎設施進度 (Core Infrastructure)

- [x] **物理與坐標標準化**：
    - 強制實施右手 Y-up 座標系（物理解算、模型、渲染全鏈路對齊）。
    - 提供 `utils/axis_converter.py` 的通用 Z-to-Y 與 Y-to-Z 轉換。
- [x] **數據探針工具 (`./inspector.py`)**：
    - 快速分析 `.npy` / `.npz` 數據維度、最大/最小值、動態軌跡分布等。
- [x] **3D 渲染與視覺化管線 (`utils/rendering/`)**：
    - `mesh_renderer.py`：使用 Pyrender 進行高精细度 3D Mesh 渲染 + 陰影貼圖 + 地板。
    - `joints_renderer.py`：基於 Matplotlib 的骨架渲染，具備 22j/24j/25j/52j 拓樸自動識別，且內置 Y-Z 自動旋轉與動態 Look-at 相機追蹤。
- [x] **動態後綴與歷史保存系統**：
    - 所有轉換器支援對檔名進行累計標記（例如：`walking_01_poses_smpl_body25.npy`），方便回溯。

---

## 3. 視覺化工具與指令 (Visualizers)

| 數據類型 | 視覺化入口腳本 | 渲染引擎細節 | 預設影片輸出路徑 |
| :--- | :--- | :--- | :--- |
| **HumanML3D (263D)** | `python visualizers/vis_humanml3d.py <file>` | 經由 22j 特徵還原器 + Matplotlib 骨架渲染 | `data/humanml3d/samples/visualizations/` |
| **OpenPose (BODY25)** | `python visualizers/vis_body25_joints.py <file>`| Matplotlib 25 關節拓樸渲染 (標準 Y-up) | `data/body25/visualizations/` |
| **Standard Joints** | `python visualizers/vis_smpl_joints.py <file>` | Matplotlib 自動匹配 17j / 22j / 24j / 52j 骨架連線 | `data/smpl_joints/samples_Xj/visualizations/` (H36M17 存於 `data/h36m17/visualizations/`) |
| **SMPL Mesh (72D)** | `python visualizers/vis_smpl_mesh.py <file>` | Pyrender + Standard SMPL 模型 FK 渲染 | `data/smpl/smpl/visualizations/` |
| **SMPL-H Mesh (156D)**| `python visualizers/vis_smplh_mesh.py <file>` | Pyrender + SMPL-H (含手部) 模型 FK 渲染 | `data/smpl/smplh/visualizations/` |

---

## 4. 骨架偵測器與工具 (Detectors & Extractors)

| # | 工具名稱 | 輸出格式 | 腳本路徑 | 進度狀態 | 負責人 | 核心技術與備註 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **RTMPose3D** | COCO-Wholebody (133j) | `detectors/rtmpose3d_detector.py` | ✅ 已完成 | Antigravity | 使用 RTMW3D-X 進行 3D 姿態提取，座標自動校正為 Y-up 公尺 |
| 2 | **HybrIK** | SMPL (24j), BODY25 (.npy) & H36M17 (.npy) | `detectors/hybrik_detector.py` | ✅ 已完成 | Antigravity | 整合 SMPL24、BODY25 與 H36M17 偵測，支援格式選擇（預設 'all'）、3D 視覺化與貼回影片（骨架/SMPL Mesh）雙重覆載渲染 |

---
| 3 | **GVHMR** | SMPL (24j), BODY25 (.npy) & H36M17 (.npy) | `detectors/gvhmr_detector.py` | ✅ 已完成 | Antigravity | 全域相機姿態感知 3D 重建，原生輸出 Y-up 物理公尺，具備 2D 骨架與 3D SMPL mesh 覆載渲染，整合 YOLOv8/ViTPose 完整前處理 |
| 4 | **BBox Preprocessor** | 2D Bounding Box (.txt) | `preprocessors/detect_bbox.py` | ✅ 已完成 | Antigravity | 統一前處理邊界框偵測器，支援 `yolov8`/`yolov11`/`rtmdet` 標準追蹤與針對滑冰影片優化的 `skater_short`/`skater_long` 演算法，支援跨模型快取共享 |

---
*最後更新時間：2026-07-21 18:40*
*維護人：Antigravity*
