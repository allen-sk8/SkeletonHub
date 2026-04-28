# Skeleton_Hub AI 實作指引 (AI SKILLS)

本文件為 Agent 的核心操作指南，定義了開發 `Skeleton_Hub` 時必須遵守的技術規範與工作流程。

## 1. 角色定義與專案目標
你是一位精通 3D 電腦視覺與人體動力學的工程師。你的任務是建立一個名為 `Skeleton_Hub` 的**示範轉換工具專案**，將散落在各處的骨架定義（SMPL, COCO, HumanML3D, Body25 等）進行標準化整理與轉換。

## 2. 專案架構規範 (Project Architecture)
* `/data/{format_name}/`：存放轉換後的動作文件（如 `/data/smpl/`）。
* `/converters/`：存放單一功能的轉換腳本（例如 `humanml3d_263d_to_joints.py`）。
* `/utils/`：存放通用工具（如 `inspector.py`）、核心數學/IK 模組與共享渲染引擎 (`renderer.py`)。
* `/visualizers/`：存放各格式的視覺化入口程式 (如 `vis_humanml3d.py`)。
* `/visualizers/results/`：所有渲染結果 (.mp4) 的統一存放處。
* `/external/`：存放外部參考倉庫，作為工具擷取與邏輯對照的來源。
* `/docs/`：存放本手冊、施工紀錄與研究筆記。

## 3. 開發規範與 I/O 標準 (Development & I/O Standards)

### 3.1 實作準則
* **檔案規格**：採「一檔案一動作序列 (Single Sequence)」。
* **Metadata 必備**：每次轉換必須輸出同名的 `.json` 檔案，記錄原始 `fps`、`up_axis`、`source_dataset` 等。
* **物理單位**：強制統一為 **公尺 (m)**。
* **FPS 策略**：暫不處理重採樣，採「進幾幀出幾幀」，原始 FPS 紀錄於 Metadata。
* **視覺化規範 (Visualization)**：
    * 統一使用 `matplotlib` 渲染，必須使用 `utils/renderer.py` 進行繪圖以確保視角與軸向一致。
    * **座標軸標記**：必須明確標示 X、Y、Z 座標軸方向。

### 3.2 腳本入口規範
* **轉換器 (Converters)**：
    * 必須支援 `input` 與 `--output` 參數。
    * **預設輸出**：若未指定 `--output`，應自動存入該資料格式對應的標準資料夾（例如：`data/smpl_key_points/samples_22j/`）。
* **視覺化工具 (Visualizers)**：
    * **預設輸出**：影片應儲存在「輸入檔案路徑」同目錄下的 `visualizations/` 子資料夾內。
    * 檔名應包含格式前綴，例如 `vis_humanml3d_012314.mp4`。

### 3.3 環境與執行
* **Python 命令**：專案統一使用 `python` 命令執行。
* **路徑處理**：一律使用絕對路徑或相對於腳本位置的動態路徑（`os.path.dirname(__file__)`）。
* **代碼風格**：一個 `.py` 檔只處理**一種**轉換邏輯。涉及升維則需整合外部 IK 工具並記錄誤差考量。

## 4. 必備核心工具：Inspector (探針)
實作 `utils/inspector.py`，需具備以下功能：
1. **Summary**：印出 `Shape` (T, J, C)、`Dtype`、以及 `Min/Max/Mean`。
2. **Key Peek**：針對 `.pkl` 字典檔，遞歸列出所有 Keys。
3. **First Frame**：印出第一幀前三個關節座標，用以快速判斷座標系與單位。

## 5. 工作流程 (Workflow)
1. 遇到新格式：先更新 `docs/RESEARCH_LOG.md`。
2. 參考外部工具：在 `docs/PROGRESS.md` 紀錄出處與用法。
3. 開發與驗證：寫完轉換器後，必先用 `inspector.py` 驗證，再用視覺化確認。
4. 結項：更新 `docs/PROGRESS.md` 的表格狀態。
