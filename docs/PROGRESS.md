# Skeleton_Hub 施工紀錄與參考日誌 (PROGRESS)

本文件紀錄專案的開發進度以及對外部工具的參考紀錄。

## 1. 施工進度表
| 來源格式 | 目標格式 | 轉換腳本 | 進度狀態 | 驗證 | 負責人 | 備註 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HumanML3D (263D)** | HumanML3D (22j) | `humanml3d_263d_to_humanml3d_22j.py` | ✅ 已完成 | ✅ 通過 | Antigravity | 支援 `./inspector.py` 驗證 |
| **HumanML3D (22j)** | HumanML3D (263D) | `humanml3d_22j_to_humanml3d_263d.py` | ✅ 已完成 | ✅ 通過 | Antigravity | 特徵提取對齊 T2M 論文 |
| **AMASS (.npz)** | SMPL-H (.pkl) | `amass_to_smplh.py` | ✅ 已完成 | ✅ 通過 | Antigravity | 保留 156D (Body+Hands) |
| **SMPL-H (.pkl)** | SMPL (24j) | `smplh_to_smpl_24j.py` | ✅ 已完成 | ✅ 通過 | Antigravity | **修正：** 已處理左右手索引偏移問題 |
| **SMPL-H (.pkl)** | HumanML3D (22j) | `smplh_to_humanml3d_22j.py` | ✅ 已完成 | ✅ 通過 | Antigravity | 專用於 T2M 模型訓練數據生成 |
| **SMPL-H (.pkl)** | SMPL-H (52j) | `smplh_to_smplh_52j.py` | ✅ 已完成 | ✅ 通過 | Antigravity | 包含完整手部指節動作 |
| **Joints (XYZ)** | SMPL-H (.pkl) | `joints_to_smplh.py` | ✅ 已完成 | ✅ 通過 | Antigravity | 支援 IK 逆向動力學擬合 |

## 2. 核心基礎設施進度
- [x] **專案結構標準化**：(Data, Converters, Utils, Visualizers)
- [x] **數據探針 (`./inspector.py`)**：支援維度分析與數值範圍檢查。
- [x] **3D 渲染引擎 (`utils/rendering/`)**：
    - `mesh_renderer.py`：Mesh 級別渲染 (Pyrender + FFmpeg)。
    - `joints_renderer.py`：關節級別渲染 (Matplotlib)，支援 **22/24/52j 自動匹配**。
- [x] **SMPL 處理核心 (`utils/smpl/handler.py`)**：支援批量 Forward Kinematics。

## 3. 視覺化工具清單
| 數據格式 | 視覺化指令 (入口腳本) | 狀態 | 輸出位置 |
| :--- | :--- | :--- | :--- |
| **HumanML3D (263D)** | `python visualizers/vis_humanml3d.py <file>` | ✅ 已完成 | 各自 `visualizations/` 目錄 |
| **Joints (22/24/52j)** | `python visualizers/vis_smpl_joints.py <file>` | ✅ 已完成 | 各自 `visualizations/` 目錄 |
| **SMPL-H (Mesh)** | `python visualizers/vis_smplh_mesh.py <file>` | ✅ 已完成 | 各自 `visualizations/` 目錄 |

---
*更新時間：2026-04-30*
