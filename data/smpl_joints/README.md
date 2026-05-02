# SMPL-Compatible 3D Keypoints Definition

本資料夾存放以 SMPL 拓撲結構為準的 3D 關節座標數據 (XYZ)。根據應用需求，主要分為 **22 節點 (HumanML3D 版)** 與 **24 節點 (標準 SMPL 版)** 兩種規格。

## 1. 骨架拓撲與關係 (Topology & Relationship)
不論是 22 還是 24 節點，其基底邏輯皆完全相同，差別僅在於「末端細節」。

![Skeleton Profile](skeleton_definition.jpg)

### 22 節點 vs 24 節點
| 規格 | 關節索引 (Indices) | 核心差異 | 主要來源 |
| :--- | :--- | :--- | :--- |
| **SMPL 24J** | 0 - 23 | 標準 SMPL 結構，包含雙手 | AMASS, SMPLify-X, MoCap 原始數據 |
| **SMPL 22J** | 0 - 21 | 去除 Index 22 & 23 (左右手掌) | HumanML3D 263D 還原, Motion Diffusion 模型 |

*   **100% 繼承**：22 節點的每一個 Index (0-21) 及其名稱，與 24 節點完全一致。
*   **Kinematic Tree**：兩者的骨架層級結構完全相容，22 節點僅是截斷了最後兩根手指末端。

## 2. 座標規範 (Coordinate Standards)
為了確保跨資料集互通性，本目錄下的數據應遵守：
*   **單位**：公尺 (Meters)。
*   **坐標系**：Y-Up (垂直向上), Z-Forward (面向前方)。
*   **歸一化 (Normalization)**：
    *   動作序列通常已將首幀對齊至原點並面向 Z+ 方向。
    *   若為原始 AMASS 提取之數據，可能保留世界坐標，需注意轉換。

## 3. 節點對照表 (Joint Map)
| ID | 名稱 | 區域 | 備註 |
| :---: | :--- | :--- | :--- |
| 0 | Pelvis (Root) | 核心 | 所有的基準點 |
| 1-2 | L/R Hips | 下肢 | |
| 4-5 | L/R Knees | 下肢 | |
| 7-8 | L/R Ankles | 下肢 | |
| 10-11 | L/R Feet | 下肢 | 22J 的末端節點 |
| 3, 6, 9 | Spine 1, 2, 3 | 軀幹 | |
| 12, 15 | Neck, Head | 頭部 | |
| 13-14 | L/R Collars | 上肢 | 鎖骨 |
| 16-17 | L/R Shoulders| 上肢 | |
| 18-19 | L/R Elbows | 上肢 | |
| 20-21 | L/R Wrists | 上肢 | 22J 的末端節點 |
| **22-23** | **L/R Hands** | **手掌** | **僅 24J 包含此二點** |

## 4. 數據儲存
*   **路徑規範**：
    *   `samples_22j/`：存放 22 關節數據。
    *   `samples_24j/`：存放 24 關節數據。
*   **維度**：`(T, 22, 3)` 或 `(T, 24, 3)`。
