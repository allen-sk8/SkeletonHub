# Standardized SMPL Parameters (.pkl)

本資料夾存放經過 **標準化處理** 後的參數化模型數據。這些數據已統一物理單位與影格率，是 `SkeletonHub` 中「參數層級」的最核心格式。

## 1. 儲存規格與分層 (Standardization)
為了支援不同的 SMPL 模型版本，資料夾採以下分層：

| 目錄 | 模型版本 | 關節數 (J) | 參數維度 (P) | 說明 |
| :--- | :--- | :---: | :---: | :--- |
| **`smpl/`** | SMPL | 24 | 72 | 最輕量的標準版本 |
| **`smplh/`** | SMPL-H | 52 | 156 | **本專案目前主力**，包含手部細節 |
| **`smplx/`** | SMPL-X | 55+ | 165+ | 包含臉部表情與手部 |

## 2. 檔案格式規範 (.pkl)
所有的參數數據一律儲存為 **Pickle (.pkl)** 格式，內部為 Python 字典，包含以下固定 Keys：

*   **`poses`**：(T, P) - `float32` 陣列，儲存旋轉角度。
*   **`trans`**：(T, 3) - `float32` 陣列，儲存全局位移。
*   **`betas`**：(16,) - `float32` 陣列，儲存體型參數。
*   **`gender`**：`str` - 'male', 'female' 或 'neutral'。
*   **`target_fps`**：`int` - 統一為 **20** (專案標準)。

## 3. 進化路徑
*   **上游來源**：由 `data/amass/` 經 `converters/amass_to_smpl.py` 產出。
*   **下游應用**：
    *   由 `converters/smpl_to_joints.py` 轉換為 **XYZ 座標**。
    *   由 `visualizers/vis_smpl.py` 進行 3D Mesh 渲染。

## 4. 視覺化結果 (Visualizations)
渲染結果（影片）應存放在本目錄下的子資料夾：
*   路徑：`data/smpl/visualizations/`
*   命名範例：`vis_smplh_walking_01.mp4`
