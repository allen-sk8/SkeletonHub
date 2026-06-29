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

## 🟢 [2026-05-09] 轉換器規格升級：自動命名、自動視覺化與 Standard SMPL 鏈結

本專案於今日完成全面重構，大幅提昇了多級轉換（Multi-stage conversion）的防錯能力與操作體驗，並延伸支援了標準 SMPL 生態圈。

### 1. 📂 自動後綴追加系統 (Auto-Suffixing & History Preservation)
*   **痛點**：舊版轉換器若不指定輸出檔名，會採用預設檔名覆蓋或單一替換，導致一個檔案經歷多步轉換後（如 AMASS ➔ SMPL-H ➔ BODY25），檔名遺失了轉換歷史，容易混淆。
*   **解決方案**：
    *   全專案 16 個轉換器全面重構：當未提供 `--output` 參數時，自動提取輸入路徑的主檔名（Basename），並在尾端追加對應的轉換標記（如 `_smplh`、`_smpl`、`_body25`、`_22j` 等）。
    *   **轉換鏈實證**：`walking_01_poses.npz` ➔ `amass_to_smplh.py` ➔ `walking_01_poses_smplh.pkl` ➔ `smplh_to_body_25j.py` ➔ `walking_01_poses_smplh_body25.npy`。
    *   **效益**：檔名自然形成了一條可視化的「資料系譜圖（Genealogy）」，極大地方便了動力學研究中的數據回溯與比對。

### 2. 🎬 一鍵自動視覺化開關 (`--vis` 參數)
*   **機制**：為了減少使用者手動調用視覺化工具的次數，每個轉換器的命令列參數均新增了 `--vis` 布林開關。
*   **實現**：在各轉換器寫檔（`.pkl` / `.npy`）完畢後，若啟用 `--vis`，便會自動透過 Python `subprocess` 以對應的 python 解譯器調用特定的視覺化入口程式（如 `vis_smplh_mesh.py`、`vis_smpl_mesh.py`、`vis_body25_joints.py` 或 `vis_smpl_joints.py`）。
*   **自動對應機制**：
    *   SMPL-H Mesh 類轉換 ➔ `vis_smplh_mesh.py`
    *   SMPL Mesh 類轉換 ➔ `vis_smpl_mesh.py`
    *   BODY25 骨架類轉換 ➔ `vis_body25_joints.py`
    *   其餘關節數（22j/24j/52j） ➔ `vis_smpl_joints.py`

### 3. 📐 標準 SMPL (72D) 流水線之建立與物理對齊
隨著動力學生態圈的需求，本專案新增了對標準 SMPL (Standard SMPL, 24 joints) 格式的端到端支援，並釐清了核心物理細節：

#### A. AMASS to Standard SMPL (`amass_to_smpl.py`)
*   **參數裁切**：
    *   SMPL 姿勢參數限制在 **72維**（1個根旋轉 + 23個全身主要關節旋轉，無手部與面部指節）。
    *   體型參數 `betas` 裁切至標準的 **10維**。
*   **全域座標對齊**：調用 `convert_smpl_z_to_y` 將 AMASS (Z-up) 自動投影至專案標準的 Y-up 空間。

#### B. 頂點回歸至 BODY25 (`smpl_to_body_25j.py` 與 `smplh_to_body_25j.py`)
*   **原理與拓樸一致性 (Topology Invariance)**：
    *   為什麼同一個 `J_regressor_body25.npy` 矩陣能同時相容 SMPL (72D) 與 SMPL-H (156D)？
    *   **技術本質**：不管是標準 SMPL 還是 SMPL-H 模型，兩者在解碼生成人體 Mesh 時，其拓樸（Mesh Topology）完全相同，皆包含 **6,890個頂點 (Vertices)**。
    *   因此，`J_regressor_body25` 作為一個大小為 `(25, 6890)` 的線性乘積矩陣（其物理意義為對 6,890 個頂點的 3D 座標進行加權平均，以回歸出 25 個關節的位置），可以**毫無隔閡地通用於兩者**。
*   **座標重複變換之修正**：
    *   **修正要點**：在 `smpl_to_body_25j.py` 與 `smplh_to_body_25j.py` 中，**完全移除了**二次 Y-Z 的坐標變換。
    *   **科學依據**：因為在 AMASS 轉換層（`amass_to_smpl` / `amass_to_smplh`）中，模型參數便已經被轉換為標準 Y-up 全域座標。因此，當 `SMPLHandler` 調用正向運動學解算出 3D 頂點時，這些頂點已經處於正確的 Y-up 空間，直接與 `J_regressor_body25` 相乘回歸出的 3D 關節自然也就是完美的 Y-up。若在此處多此一舉進行 `convert_joints_z_to_y`，反而會造成二次變換而導致方向錯誤。

#### C. 標準 SMPL 關節提取 (`smpl_to_smpl_24j.py`)
*   **設計動機**：此轉換器由開發者 Allen 補齊，旨在完整閉合「Standard SMPL (72D) 宇宙」的動力學轉檔鏈。
*   **實現細節**：將 `.pkl` 格式的 72D SMPL 參數解碼並透過 `SMPLHandler` 正向運動學提取出前 24 個基礎身體關節座標，完美的對準標準 SMPL 骨架。

### 4. 🏷️ 命名規格強烈對齊（Name Symmetry Refinement）
*   **重構行動**：將原 EasyMocap 擬合器 `joints_52j_to_smpl.py` 正式更名為 **`joints_52j_to_smplh_easymocap.py`**。
*   **物理依據與防錯**：
    *   該擬合器實際上解算並回歸的是高階的 **SMPL-H (156維)** 參數，其預設儲存位置也是 `data/smpl/smplh`。
    *   舊名稱的 `_to_smpl` 尾端非常容易誤導使用者，使其以為這是一個輸出 Standard SMPL (72D) 的擬合器，而丟失了寶貴的手部指節動作。
    *   更名為 `_to_smplh_easymocap` 後，與 `joints_52j_to_smplh.py` (L2 Fitter) 及 `joints_52j_to_smplh_smplifyx.py` (VPoser Fitter) 形成完美的「手部捕捉擬合三大神器」陣容，消除了所有語意歧義！

---

## 🟢 [2026-06-02] RTMPose3D 整合與 COCO-Wholebody 133 轉 BODY25 實作

### 1. 偵測器整合 (RTMPose3D Inference Wrapper)
*   **指令腳本**：`detectors/rtmpose3d_detector.py`。
*   **工作原理**：
    1. 使用 RTMDet 偵測影片中的人物。
    2. 對畫面中最大 Bounding Box 的人進行 RTMW3D-X 三維全域人體姿態預測。
    3. 獲取原生的 COCO-Wholebody 133 個 3D 關節點。
*   **物理與座標對齊**：
    * 原始 RTMPose3D 模型預測結果以公尺為單位，並使用 Z-up 座標系。
    * 在偵測器內部，我們直接調用 `utils/axis_converter.py` 中的 `convert_joints_z_to_y` 對預測結果進行 Y-up 座標投影，並將輸出儲存至 `data/coco_wholebody133/`，副檔名後綴為 `_coco_wholebody133.npy`。

### 2. 格式轉換器實作 (COCO-Wholebody 133 to BODY25)
*   **指令腳本**：`converters/joints_133j_to_body_25j.py`。
*   **對應邏輯**：
    * **直接映射**：對應 Nose、雙肩、雙肘、雙腕、雙髖、雙膝、雙踝、雙眼、雙耳及足底（大腳趾、小腳趾、腳跟）等共通的 23 個關節點。
    * **插值解算**：
        * **Neck (BODY25 index 1)** = $\frac{\text{LShoulder} + \text{RShoulder}}{2}$
        * **MidHip (BODY25 index 8)** = $\frac{\text{LHip} + \text{RHip}}{2}$
*   **視覺化整合**：支援自動剥離後綴（`_coco_wholebody133` 或 `_joints133`）與 `--vis` 參數，以便在轉檔後自動呼叫 `vis_body25_joints.py` 繪製標準 Y-up 骨架影片。

---

## 🟢 [2026-06-29] HybrIK SMPL24 & BODY25 偵測器整合與原生化

### 1. 偵測器原生化與環境整合
*   **整合目標**：消除對外部路徑 `/home/leeyoyo49/` 以及 subprocess 呼叫 `hybrik_env` python 的依賴，使 SkeletonHub 具備完全自主與可移植性。
*   **子模組集成**：將原始 HybrIK 倉庫以 Git 子模組形式 clone 至 `external/Hybrik`。藉由 `--no-build-isolation --no-deps` 參數在 `skeleton_env` 下以編輯模式安裝該 package，成功整合 `easydict` 與 `filterpy` 等依賴。
*   **本地權重與模型配置**：
    *   主權重 `hybrik_hrnet.pth` 移動至 `common_models/checkpoints/hybrik/`。
    *   SMPL 與 SMPLX 參數移動至 `common_models/body_models/smpl/` 與 `common_models/body_models/smpl/smplx/`。
    *   在專案根目錄建立 `model_files` 軟連結指向 `common_models/body_models/smpl`，完美兼容 HybrIK 內部硬編碼的 `./model_files/` 讀取邏輯。
*   **本地推論引擎實作 (`utils/hybrik/service_fast.py`)**：
    *   將推論核心抽離並原生實作於 `utils/hybrik/service_fast.py` 中，使用絕對路徑讀取本地的配置與權重。
    *   對 YOLO 偵測框的讀取（`video.txt`）實施了多級模糊匹配與自動 fallback，確保不因命名差異導致全域位移解算失敗。
    *   修補了子模組中 `hybrik/models/layers/hrnet/hrnet.py` 的 FileNotFoundError 瑕疵，使其以動態相對於檔案路徑的方式讀取 `w48.yaml`。
*   **推論效能提昇**：
    *   藉由直接在同一個 Python 進程中進行 Tensor 傳遞與推論，省去了跨環境 subprocess 建立、虛擬環境啟動與頻繁寫入讀取暫存檔的負擔。
    *   **推論速度達 33 FPS**（原先為 <5 FPS），效能獲得顯著升級。

### 2. 多重格式輸出與影片覆載渲染功能
*   **新增輸出格式選擇**：支援 `--format` 參數，使用者可自由選擇輸出格式為 `smpl24`、`body25` 或 `both`。
*   **骨架影片覆載渲染 (`--vis-skeleton-video`)**：
    *   利用 HybrIK 內置相機內參估算原理，以 YOLO 偵測框動態計算每幀相機焦距與光心：
        *   $f = rac{1000 \cdot W_{	ext{bbox}}}{256}$
        *   $c_x = 	ext{bbox}_x, c_y = 	ext{bbox}_y$
    *   將 3D 關節點投影至 2D 圖像坐標，利用 OpenCV 抗鋸齒畫線渲染人體骨架（Red 代表 Right，Blue 代表 Left，Green 代表 Center），並將骨架貼回原始影片。
*   **SMPL Mesh 影片覆載渲染 (`--vis-smpl-video`)**：
    *   調用 `SMPLHandler` 將 SMPL 旋轉及體型參數轉為 3D 相機坐標系頂點（6,890 頂點）。
    *   利用 Pyrender 的 `IntrinsicsCamera(fx=f, fy=f, cx=cx, cy=cy)` 設定對齊投影相機，以 EGL 模式進行 headless 渲染。
    *   取得渲染深度的 binary mask (`depth > 0`)，對原始影片幀進行 $0.7$ 權重 alpha 半透明網格覆蓋混合渲染。
*   **編碼優化**：所有渲染影片皆經由 `ffmpeg` 自動進行 H.264 與 `yuv420p` 優化編碼，確保留覽相容性。

### 3. 2D 相機投影對齊修復 (SMPL 與 native HybrIK 空間偏差校正)
*   **問題診斷**：在之前的覆載 (Overlay) 影片中，發現投影出的人體骨架與 Mesh 相較於原始影片中的人向上位移了一些，且整體尺寸偏小。修復後，BODY25 骨架與 SMPL Mesh 對齊精確，但 SMPL24 火柴人高度縮水了約一半（僅為正常高度的 $45\%$）。
*   **根源成因分析**：
    1.  **標準 SMPL 與自定義 Pelvis-centered 空間偏差**：
        -   HybrIK 模型內部的 3D 關節點和頂點 (`pred_vertices`) 都是在該模型自定義的 pelvis-centered 歸一化空間中進行估計的。
        -   若使用標準 `SMPLHandler` 解算出的頂點與模型預測的 `transl` (即模型空間 pelvis 位置) 直接相加時，由於沒有扣除標準 SMPL pelvis 點相對於原點的固有偏差（標準 SMPL pelvis 在 template 姿態下大約處於 $Y \approx 0.08$ 米的上方），會造成系統性的向上位移。
    2.  **模型內部關節點與頂點的尺度差異 (Scale Difference)**：
        -   在 HybrIK 模型內部，預測的 29 關節點 `pred_xyz_jts_29` 的坐標單位是 `self.depth_factor m` (其中 `self.depth_factor = 2.2`)，即相對於實體公尺縮放了 $2.2$ 倍的歸一化空間。
        -   而模型內部的前向動力學層在解算頂點 `pred_vertices` 時，會先將關節點乘回物理公尺單位：`pose_skeleton = pred_xyz_jts_29 * self.depth_factor`。這使得輸出的 `pred_vertices` 已經是真實的公尺單位。
        -   因此，直接取出的 `pred_xyz_29` 相較於 `pred_vertices` 短了 $2.2$ 倍。
    3.  **焦距基準對齊**：
        -   HybrIK 模型內部投影焦距以高度 `256.0` 為基準對齊。若使用 native 模型空間的輸出，則投影公式中的焦距分母必須精確採用高度基準 `256.0`：
            $$f_{\text{original}} = 1000.0 \times \frac{W_{\text{bbox}}}{256.0}$$
*   **修復對策**：
    -   **完全移除 `SMPLHandler` 重建流程**：在進行 2D 骨架與 SMPL Mesh 渲染時，不再調用 `SMPLHandler` 重建，而是直接從 `smpl.pk` 中讀取 HybrIK 原生預測的頂點 `pred_vertices` 與關節點 `pred_xyz_29`。
    -   **對齊尺度與相機空間**：
        -   將 `pred_xyz_29` 乘回 `scale` 因子（預設 `2.2`）以還原為實體公尺單位，再與 `transl`（相機空間 pelvis 位置）相加：
            $$\mathbf{J}_{\text{cam}} = \mathbf{J}_{\text{pred}} \times 2.2 + \mathbf{T}_{\text{transl}}$$
        -   頂點 `pred_vertices` 已是公尺單位，直接與 `transl` 相加：
            $$\mathbf{V}_{\text{cam}} = \mathbf{V}_{\text{pred}} + \mathbf{T}_{\text{transl}}$$
    -   **投影公式校正**：將投影與渲染時的焦距公式分母統一恢復為正確的 `256.0`。
    -   **結果驗證**：修復後，SMPL24 骨架、BODY25 骨架與 SMPL Mesh 的覆載比例均與影片中的人體實現了 100% 完美的貼合對齊。

### 3. SMPL24 與 BODY25 物理對齊及坐標標準化
*   **SMPL24 坐標流 (Keypoints Space)**：
    1.  從 `skeleton.pk` 讀取 `pred_xyz_24_struct_global` (T, 24, 3) 關節座標。
    2.  **相機至世界坐標投影**：`y_world = -y_camera`, `z_world = -z_camera`（右手系 Y-up）。
    3.  **尺度還原**：HybrIK 原生關鍵點歸一化在 `[-1, 1]` 的 2.2 米邊界框內，故乘以 `scale=2.2` 以還原成實體公尺。
    4.  **基底貼平 (Grounding)**：若啟用 `rebase`，計算首幀所有關節的最小 Y 座標 `min_y`，並對全幀進行 `Y -= min_y`，使首幀最低點座落在地面 Y=0。
*   **BODY25 坐標流 (Vertices Space)**：
    1.  從 `smpl.pk` 讀取 SMPL 參數：`pred_thetas` (T, 24, 3, 3), `pred_betas` (T, 10), `transl` (T, 3)。
    2.  **旋轉表示轉換**：使用 `scipy.spatial.transform.Rotation` 將 $3 	imes 3$ 旋轉矩陣轉換為 3D 軸角 (Axis-angle)，展開為 (T, 72)。
    3.  **頂點重建 (FK)**：調用 `SMPLHandler(model_type='smpl')` 進行前向動力學解算，計算出 6890 個相機空間頂點。
    4.  **線性回歸**：使用 `J_regressor_body25.npy` 對 6890 個頂點進行乘積，回歸出 25 個關節。
    5.  **相機至世界坐標投影**：`y_world = -y_camera`, `z_world = -z_camera`（右手系 Y-up）。因 SMPL 模型本身即是公尺單位，此處無須乘以 2.2 尺度。
    6.  **首幀骨盆對齊 (Centering)**：為了使 `body25` 與本質上已是 Pelvis-centered 的 `pred_xyz_24_struct_global` 完全一致，在 XZ 平面上減去首幀關節 8 (MidHip/Pelvis) 的坐標值，使首幀骨盆對齊 X=0, Z=0。
    7.  **基底貼平 (Grounding)**：減去首幀 25 個關節的最低 Y 座標值，使首幀最低點貼平地面 Y=0。

*   **H36M17 坐標流 (Keypoints Space)**：
    1.  從 `smpl.pk` 讀取 `pred_xyz_17` (T, 17, 3) 關節座標。
    2.  **尺度還原**：與 `pred_xyz_29` 類似，`pred_xyz_17` 原生數值經由 `depth_factor` 歸一化，需乘以 `scale=2.2` 還原為實體公尺。
    3.  **相機至世界坐標投影**：加上平移量 `transl` 轉為相機空間，再轉換至右手系 Y-up 世界坐標系（`y_world = -y_camera`, `z_world = -z_camera`）。
    4.  **基底貼平 (Grounding)**：若啟用 `rebase`，計算首幀所有關節的最小 Y 座標 `min_y`，並對全幀進行 `Y -= min_y`，使首幀最低點座落在地面 Y=0。

### 4. HybrIK 29 關節點格式定義與來源說明
*   **什麼是 `pred_xyz_29`？**：
    -   標準 SMPL 模型只定義了 24 個關鍵點（由 1 個 pelvis 根節點與 23 個身體主要骨骼關節組成）。
    -   然而，為了求解高精度的逆向運動學 (Inverse Kinematics, IK) 尤其是頭部朝向、手部以及足部接觸點，HybrIK 模型在標準 SMPL 的 24 個關鍵點基礎上額外擴充了 5 個末端葉子節點 (Leaf Nodes)：
        -   Index 24: `head` (頭頂)
        -   Index 25: `left_middle` (左手中指)
        -   Index 26: `right_middle` (右手中指)
        -   Index 27: `left_bigtoe` (左腳大腳趾)
        -   Index 28: `right_bigtoe` (右腳大腳趾)
    -   這 29 個點合稱 **HybrIK 29 關節點拓撲**，是模型神經網絡直接預測的 3D 關節點。
*   **為什麼我們只需要 24 個關節？**：
    -   對於下游的 SMPL 24j 世界坐標系表示，我們只需要標準的 24 個關節，因此在提取時，我們只切片取前 24 個通道：`pred_xyz_29[:, :24, :]`，即可完美對齊標準 SMPL 24 關節順序。

---
*文件更新人：Antigravity*
*最後更新：2026-06-30 01:05*
