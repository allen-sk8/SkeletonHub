# Standardized SMPL Parameters (.pkl)

本資料夾存放經過 **標準化處理** 後的參數化模型數據。這些數據已統一物理單位與影格率，是 `SkeletonHub` 中「參數層級」的最核心格式。

## 1. SMPL 家族變體關係 (Variants Relationship)
了解不同版本的差異對於正確載入模型至關重要：

| 版本 | 全稱 | 關節數 | 包含內容 | 特點與用途 |
| :--- | :--- | :---: | :--- | :--- |
| **SMPL** | Standard | 24 | 身體 (Body) | 2015 年發佈，最基礎的火柴人架構。 |
| **SMPL-H** | + Hands | 52 | 身體 + 雙手 (MANO) | **AMASS 資料集的主力**，修正了手部僵硬問題。 |
| **SMPL-X** | eXpressive | 55+ | 身體 + 雙手 + 臉部 | **2019 年最新標準**，包含 FLAME 臉部表情，目前研究的首選。 |

**進階關係：**
*   **向下相容**：SMPL-X 的參數集可以透過特定轉換表對應回 SMPL-H 或 SMPL。
*   **數據演進**：AMASS 雖然多數是 SMPL-H，但現在已有工具將其全面升級至 SMPL-X 空間。

## 2. 目錄分層規範 (Directory Layering)
為了支援不同的模型版本，資料夾採以下分層：

*   **`smpl/`**：標準 SMPL (72 參數)。
*   **`smplh/`**：SMPL + Hands (156 參數)。
*   **`smplx/`**：SMPL + Hands + Face (165+ 參數)。

## 3. 檔案格式規範 (.pkl)
所有的參數數據一律儲存為 **Pickle (.pkl)** 格式，內部為 Python 字典，包含以下固定 Keys：

*   **`poses`**：(T, P) - `float32` 陣列，儲存旋轉角度。
*   **`trans`**：(T, 3) - `float32` 陣列，儲存全局位移。
*   **`betas`**：(16,) - `float32` 陣列，儲存體型參數。
*   **`gender`**：`str` - 'male', 'female' 或 'neutral'。
*   **`target_fps`**：`int` - 統一為 **20** (專案標準)。

## 4. 進化路徑
*   **上游來源**：由 `data/amass/` 經 `converters/amass_to_smplh.py` 產出。
*   **下游應用**：
    *   由 `converters/smpl_to_joints.py` 轉換為 **XYZ 座標**。
    *   由 `visualizers/vis_smpl.py` 進行 3D Mesh 渲染。

## 5. 視覺化結果 (Visualizations)
渲染結果（影片）預設存放在對應規格資料夾下的 `visualizations/` 子目錄中。
例如：`data/smpl/smplh/visualizations/`
