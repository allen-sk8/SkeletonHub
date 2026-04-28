# Skeleton_Hub 施工紀錄與參考日誌 (PROGRESS)

本文件紀錄專案的開發進度以及對外部工具的參考紀錄。

## 1. 施工進度表
| 來源格式 | 目標格式 | 轉換腳本狀態 | Inspector 驗證 | 負責人 | 備註 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HumanML3D (263D) | Joints (22j) | `humanml3d_263d_to_joints.py` | ✅ 已完成 | ✅ 通過 | Antigravity |
| Joints (22j) | HumanML3D (263D) | `humanml3d_joints_to_263d.py` | ✅ 已完成 | ✅ 通過 | Antigravity |
| AMASS (.npz) | SMPL (.pkl) | `amass_to_smpl.py` | ✅ 已完成 | ✅ 通過 | Antigravity |
| SMPL (.pkl) | Joints (24j) | `smpl_to_joints.py` | 📝 待實作 | ➖ | Antigravity |

## 2. 核心規範進度
- [x] 建立專案基礎結構 (Data, Converters, Utils, Visualizers)
- [x] 建立 `utils/inspector.py` (數據探針)
- [x] 建立 `utils/renderer.py` (3D 動作渲染器)
- [x] 實施 **I/O 標準化規範** (自動導向輸出與視覺化資料夾)
- [x] 建立 **HumanML3D 資料集規格書** 與生成工具

## 3. 視覺化工具進度
| 格式 | 入口腳本 | 狀態 | 輸出路徑 |
| :--- | :--- | :--- | :--- |
| HumanML3D (263D) | `vis_humanml3d.py` | ✅ 已完成 |各自data資料夾下的visualizations資料夾 |
| Raw Joints (T,J,3) | `vis_joints.py` | ✅ 已完成 |各自data資料夾下的visualizations資料夾 |
| SMPL Parameters | `vis_smpl.py` | 📝 待實作 |各自data資料夾下的visualizations資料夾 |

## 2. 外部參考工具紀錄 (External Tool Log)
凡參考 `/external/` 中的倉庫或網路現有工具時，需在此詳實紀錄。

| 工具名稱 | 來源 (URL/Path) | 用途 | 如何使用 / 關鍵邏輯 | 擷取內容 |
| :--- | :--- | :--- | :--- | :--- |
| SMPLify-X | [GitHub](https://github.com/vchoutas/smplify-x) | 2D/3D to SMPL | 透過 IK 優化最小化重投影誤差 | 參考其 Regressor 權重 |
| HumanML3D | `external/HumanML3D/motion_representation.ipynb` | **263D 特徵提取流水線** | **1. 座標對齊**：減去根節點 XZ 座標並繞 Y 軸旋轉使首幀朝向 Z+ (`get_rifke`)。<br>**2. 特徵拼接**：依序拼接 Root (4D), RIC (63D), Rot (126D), Vel (66D), Contacts (4D)。<br>**3. 腳底接觸**：根據關節速度閾值 (0.002) 判定。 | `process_file`, `get_cont6d_params`, `foot_detect` 等函數邏輯 |
| HumanML3D | `external/HumanML3D/paramUtil.py` | **骨架拓樸定義** | 定義了 T2M (22J) 的運動鏈與關節索引 (如 `FACE_JOINT_INDX=[2, 1, 17, 16]`)。 | `t2m_kinematic_chain`, `t2m_raw_offsets` |

## 3. 已移植之外部庫 (Ported Libraries)
為了確保專案的獨立性與可移植性，以下代碼已從 `external/` 移植至 `utils/humanml3d_lib/` 並修正了 NumPy 1.24+ 相容性：

| 模組名稱 | 原始路徑 | 移植路徑 | 修正內容 |
| :--- | :--- | :--- | :--- |
| `quaternion` | `external/HumanML3D/common/quaternion.py` | `utils/humanml3d_lib/quaternion.py` | 修正 `np.float` 為 `float` |
| `skeleton` | `external/HumanML3D/common/skeleton.py` | `utils/humanml3d_lib/skeleton.py` | 修正相對匯入路徑 |
| `paramUtil` | `external/HumanML3D/paramUtil.py` | `utils/humanml3d_lib/paramUtil.py` | 提取 22 關節定義與 Offsets |


## 4. 系統環境設定 (Environment)
- [x] 建立 Conda 環境 `skeleton_env` (Python 3.10)
- [x] 安裝核心依賴 (numpy<2, matplotlib, torch, scipy, tqdm)
- [x] 修正 Matplotlib 3D 渲染 Bug (Axes3D.lines setter issue)

