# SkeletonHub 研究與實作日誌 (RESEARCH_LOG)

本文件詳實紀錄各項技術決策、轉換邏輯、代碼來源以及物理參數規範，作為專案的「技術聖經」。

---

## 🟢 [2026-04-30] SMPL 家族核心技術規範

### 1. 座標系與物理標準 (Physical Standards)
為了與現代渲染引擎（如 Unity, Pyrender）及主流數據集（HumanML3D）對齊，本專案強制執行以下標準：
*   **座標系**：右手坐標系 (Right-handed System)。
*   **向上軸 (Up-axis)**：嚴格統一為 **Y-up** (Y-in, Y-out)。所有外部 Z-up 數據皆需先透過 `utils/axis_converter.py` 轉換。
*   **物理單位**：公尺 (Meters, m)。
*   **旋轉表示**：
    *   外部存儲：軸角 (Axis-angle, 3D) 或四元數 (Quaternion, 4D)。
    *   運算內部：連續型 6D 旋轉 (Continuous 6D Rotation) 以利神經網路收斂。

---

### 2. 轉換器技術拆解 (Converter Deep Dive)

#### A. AMASS to SMPL-H (`amass_to_smplh.py`)
*   **參考來源**：參考了 AMASS 官方數據讀取腳本與 `smplx` 庫的 `body_models` 加載邏輯。
*   **數據流**：
    1.  從 `.npz` 讀取 `poses` (N, 156), `betas` (16,), `trans` (N, 3), `mocap_frame_rate`。
    2.  **維度補齊**：若 `betas` 只有 10 維，則補零至 16 維，確保與 SMPL-H 權重對齊。
    3.  **格式打包**：輸出為 `.pkl` 字典，包含 `gender` 字段（必備，因 SMPL-H 區分男女模型）。
*   **維度細節**：SMPL-H (156D) = Root (3) + Body (21x3=63) + L_Hand (15x3=45) + R_Hand (15x3=45)。

#### B. SMPL-H to Joint Series (`smplh_to_..._j.py`)
*   **核心組件**：`utils/smpl/handler.py` 中的 `SMPLHandler` 類別。
*   **FK 邏輯來源**：調用 `smplx.create(...)` 生成模型對象，利用其 `forward()` 函數執行前向動力學。
*   **座標變換 (Coordinate Transformation)**：
    *   原始模型輸出：AMASS 座標系 (Z-up)。
    *   轉換代碼：
        ```python
        # AMASS Z-up -> Y-up
        joints_y_up[:, :, 1] = joints[:, :, 2]
        joints_y_up[:, :, 2] = -joints[:, :, 1]
        ```
*   **骨架對齊矩陣 (Skeleton Mapping)**：
    | 目標 | 索引選取邏輯 | 參考出處 |
    | :--- | :--- | :--- |
    | **22j (H3D)** | `[:22]` | `external/HumanML3D/common/skeleton.py` |
    | **24j (SMPL)**| `[:22]` + `[22]`(L) + `[37]`(R) | 修正 naive slice `[:24]` 導致的手部索引偏移錯誤。 |
    | **52j (H)** | `[:52]` | SMPL-H 官方關節拓樸。 |

---

### 3. HumanML3D 特徵流水線技術細節

#### A. 263D 特徵維度定義
本專案嚴格遵循 `HumanML3D` 論文定義的特徵空間：
1.  **Root Velocity (4D)**：[r_vel_y, l_vel_x, l_vel_z, root_height]。
2.  **RIC (Local Positions, 63D)**：21 個關節相對於根節點的局部座標 (21 * 3)。
3.  **Rotation (126D)**：21 個關節的 6D 旋轉表示 (21 * 6)。
4.  **Linear Velocity (66D)**：22 個關節在局部座標系下的線速度 (22 * 3)。
5.  **Foot Contacts (4D)**：[L_Heel, L_Toe, R_Heel, R_Toe]。

#### B. 核心演算邏輯來源
*   **旋轉對齊 (`get_rifke`)**：參考 `external/HumanML3D/motion_representation.ipynb` 中的座標對齊函數。邏輯為：將每一幀的根節點位置平移至原點，並繞 Y 軸旋轉使得首幀朝向 Z 軸正方向。
*   **連續 6D 旋轉轉換**：參考 `utils/humanml3d/lib/quaternion.py` 中的 `quaternion_to_cont6d` 函數，將四元數轉為矩陣的前兩列。
*   **腳底接觸偵測**：參考 `utils/humanml3d/utils.py` 中的 `foot_detect`，利用關節在相鄰幀間的位移量（閾值 0.002m）判定是否接觸地面。

---

## 🟢 [2026-04-28] 早期施工紀錄 (存檔)
*(此處保留原有的早期紀錄以維護文件完整性)*
- 建立專案基礎結構 (Data, Converters, Utils, Visualizers)。
- 實施 I/O 標準化規範。
- 建立 `common_models` 外部權重管理中心。

---

## 🟢 [2026-05-02] EasyMocap 高精度擬合流水線整合

### 1. 轉接器實作 (EasyMocap Wrapper)
為解決舊版 L2 擬合缺乏解剖學先驗的問題，引入 EasyMocap 作為核心優化引擎：
*   **組件**：`utils/smpl/easymocap_wrapper.py`
*   **技術突破**：
    *   **強制關節回傳**：實作 `SMPLModelWrapper` 強制設定 `return_smpl_joints=True`，解決了官方庫在優化時隱藏基礎 24 關節導致 Loss 為零的問題。
    *   **坐標轉換 (Y-to-Z)**：實作 `(x, y, z) -> (x, -z, y)` 轉換，確保輸入關節與 EasyMocap 內部先驗 (AMASS Standard) 對齊。

### 2. 環境相容性補丁 (Legacy Library Support)
針對 Python 3.11+ 與現代 Numpy 環境，對 `chumpy` 進行了熱補丁處理：
*   **API 修復**：`inspect.getargspec` -> `inspect.getfullargspec`。
*   **類型修復**：修正 `numpy.bool`, `numpy.int`, `numpy.float` 等過時類型引用。

### 3. 渲染管線升級 (Standardized Axis Conversion)
*   **架構統一**：廢棄各模組內的手寫轉換邏輯，統一調用 `utils/axis_converter.py`。
    *   **擬合前 (Y-to-Z)**：`convert_joints_y_to_z`。
    *   **渲染前 (Z-to-Y)**：`convert_joints_z_to_y`。
*   **動態相機**：`MeshRenderer` 實作動態 **Look-at** 邏輯，自動追蹤人物中心點並調整焦距與視角。

## 🟢 [2026-05-02] 52j (SMPL-H) 高精度擬合與手部捕捉

### 1. SMPL-H 擬合流水線
*   **指令腳本**：`converters/joints_52j_to_smpl.py`。
*   **手部優化**：解鎖了 156 維參數空間，引入 `k3d_hand` 能量項，成功捕捉手指細微動作。
*   **性能**：181 幀數據在 LBFGS 1000 次疊代下表現穩定，手部 Loss 降幅達 99.9%。

### 2. 模型加載強韌化 (SMPLHandler Upgrade)
*   **自動回溯邏輯**：
    1.  優先嘗試 `.pkl` 格式。
    2.  若 `neutral` 缺失，自動按順序嘗試 `female` -> `male`。
    3.  支援 `.npz` 格式作為備選，確保在不同版本模型庫間的相容性。

---
*文件更新人：Antigravity*
*最後更新：2026-05-02 11:50*
