# Skeleton_Hub 施工紀錄與參考日誌 (PROGRESS)

本文件紀錄專案的開發進度以及對外部工具的參考紀錄。

## 1. 施工進度表
| 來源格式 | 目標格式 | 轉換腳本狀態 | Inspector 驗證 | 負責人 | 備註 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HumanML3D (263D) | Joints (22j) | `humanml3d_263d_to_joints.py` | ✅ 已完成 | ✅ 通過 | Antigravity |
| SMPL (.pkl) | Joints (24j) | `smpl_to_joints.py` | 📝 待實作 | ➖ | Antigravity |
| AMASS (.npz) | SMPL (.pkl) | `amass_to_smpl.py` | 📝 待實作 | ➖ | Antigravity |

## 2. 核心規範進度
- [x] 建立專案基礎結構 (Data, Converters, Utils, Visualizers)
- [x] 建立 `utils/inspector.py` (數據探針)
- [x] 建立 `utils/renderer.py` (3D 動作渲染器)
- [x] 實施 **I/O 標準化規範** (自動導向輸出與視覺化資料夾)
- [x] 建立 **HumanML3D 資料集規格書** 與生成工具

## 3. 視覺化工具進度
| 格式 | 入口腳本 | 狀態 | 輸出路徑 |
| :--- | :--- | :--- | :--- |
| HumanML3D (263D) | `vis_humanml3d.py` | ✅ 已完成 | `/visualizers/results/` |
| Raw Joints (T,J,3) | `vis_joints.py` | ✅ 已完成 | `/visualizers/results/` |
| SMPL Parameters | `vis_smpl.py` | 📝 待實作 | `/visualizers/results/` |

## 2. 外部參考工具紀錄 (External Tool Log)
凡參考 `/external/` 中的倉庫或網路現有工具時，需在此詳實紀錄。

| 工具名稱 | 來源 (URL/Path) | 用途 | 如何使用 / 關鍵邏輯 | 擷取內容 |
| :--- | :--- | :--- | :--- | :--- |
| SMPLify-X | [GitHub](https://github.com/vchoutas/smplify-x) | 2D/3D to SMPL | 透過 IK 優化最小化重投影誤差 | 參考其 Regressor 權重 |
| HumanML3D | [GitHub](https://github.com/EricGuo551k/humanml3d) | Motion Features | 提取 263D 特徵向量 | 參考其特徵組成與單位規範 |

## 3. 已移植之外部庫 (Ported Libraries)
為了確保專案的獨立性與可移植性，以下代碼已從 `external/` 移植至 `utils/humanml3d_lib/` 並修正了 NumPy 1.24+ 相容性：

| 模組名稱 | 原始路徑 | 移植路徑 | 修正內容 |
| :--- | :--- | :--- | :--- |
| `quaternion` | `external/HumanML3D/common/quaternion.py` | `utils/humanml3d_lib/quaternion.py` | 修正 `np.float` 為 `float` |
| `skeleton` | `external/HumanML3D/common/skeleton.py` | `utils/humanml3d_lib/skeleton.py` | 修正相對匯入路徑 |
| `paramUtil` | `external/HumanML3D/paramUtil.py` | `utils/humanml3d_lib/paramUtil.py` | 提取 22 關節定義與 Offsets |
| `profiler` | `utils/skeleton_profiler.py` (已遷移) | `data/humanml3d/scripts/generate_profile.py` | 客製化 2D 投影規格圖生成 |

## 4. 系統環境設定 (Environment)
- [x] 建立 Conda 環境 `skeleton_env` (Python 3.10)
- [x] 安裝核心依賴 (numpy<2, matplotlib, torch, scipy, tqdm)
- [x] 修正 Matplotlib 3D 渲染 Bug (Axes3D.lines setter issue)

