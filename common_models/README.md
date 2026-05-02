# Common Models (Weights) 管理規範

本目錄存放所有外部動捕模型與人體建模的權重檔案。為了確保代碼相容性與來源追溯，請嚴格遵守以下結構。

## 1. 目錄結構與下載來源

### A. SMPL (Standard)
*   **來源**: [MPI SMPL Website](https://smpl.is.tue.mpg.de/)
*   **內容**: 基礎 24 關節人體模型（不含手部動作）。
*   **存放路徑**:
    ```text
    smpl/
    ├── SMPL_MALE.pkl
    ├── SMPL_FEMALE.pkl
    └── SMPL_NEUTRAL.pkl
    ```

### B. SMPL-H (用于 AMASS)
*   **來源**: [MANO Website](https://mano.is.tue.mpg.de/)
*   **內容**: 包含手部 (MANO) 的完整模型。
*   **重點**: 請下載 **"SMPLH model version ready to load by the smplx python package"**。
*   **存放路徑**:
    ```text
    smplh/
    ├── SMPLH_MALE.pkl        # 庫調用核心 (smplx 專用)
    ├── SMPLH_FEMALE.pkl      # 庫調用核心 (smplx 專用)
    ├── male/model.npz        # AMASS 原始權重備份
    ├── female/model.npz      # AMASS 原始權重備份
    └── neutral/model.npz     # AMASS 原始權重備份
    ```

### C. SMPL-X (Expressive)
*   **來源**: [SMPL-X Website](https://smpl-x.is.tue.mpg.de/)
*   **內容**: 包含臉部表情、身體與手部的最新標準。
*   **存放路徑**:
    ```text
    smplx/
    ├── male/
    ├── female/
    └── neutral/
    ```

### D. VPoser (Body Pose Prior)
*   **來源**: [SMPL-X Website](https://smpl-x.is.tue.mpg.de/download.php)
*   **內容**: 包含臉部表情、身體與手部的最新標準。
*   **存放路徑**:
    ```text
    vposer_v1_0/
    ```

## 2. 代碼調用規範
在專案中使用 `utils/smpl_handler.py` 進行調用，該工具會自動根據 `model_type` 前往對應資料夾加載：

```python
from utils.smpl_handler import SMPLHandler
handler = SMPLHandler(model_type='smplh') # 自動指向 smplh/ 目錄
```

## 3. 版本控制注意事項
*   本目錄下的所有模型權重（`.pkl`, `.npz`, `.ckpt`）預設已被 `.gitignore` 排除，**禁止上傳至 Git**。
*   僅保留 `README.md` 以供參考。
