# AMASS Raw Data Seeds (Source)

本資料夾存放從 [AMASS](https://amass.is.tue.mpg.de/) 官網下載的最原始數據種子。這些數據是尚未經過任何處理（如影格率轉換、座標系對齊）的「原始檔案」。

## 1. 資料來源與下載規格
*   **建議規格**：下載時請務必選擇 **SMPL+H G** (包含手部與 DMPL)。
*   **資料特點**：AMASS 已經將各種不同來源的動捕數據（如 CMU, KIT）統一擬合至 SMPL+H 模型，但仍保留了原始的採樣頻率 (FPS) 與世界座標系。

## 2. 目錄結構
為了保持專案整潔，原始檔案應放置於對應規格的子資料夾中：

*   **`samples_smpl_H_G/`**：存放下載的 `.npz` 檔案。建議保留原始資料集的目錄樹（例如 `SSM_synced/...`）。

## 3. 原始數據規格 (Raw NPZ Keys)
標準的 AMASS `.npz` 包含以下核心欄位：
*   **`poses`**：(T, 156) - 包含 Root Orient (3), Body Pose (63), Hand Pose (90)。
*   **`trans`**：(T, 3) - 全局座標位移。
*   **`betas`**：(16,) - 人物體型參數。
*   **`gender`**：性別（影響 SMPL 模型選擇）。
*   **`mocap_framerate`**：原始動捕設備採樣率。

## 4. 進化流 (Evolution Workflow)
此處的數據不應直接用於渲染或訓練，必須經過以下流程：
1.  **原始輸入**：`data/amass/samples_smpl_H_G/*.npz`
2.  **轉換操作**：執行 `converters/amass_to_smpl.py`。
3.  **標準產出**：儲存至 `data/smpl/smplh/*.pkl` (統一為 20 FPS 並精簡欄位)。

---
> **注意**：由於原始數據通常體積極大，此資料夾下的 `.npz` 檔案預設已被加入 `.gitignore` 排除。
