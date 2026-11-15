# 3D 關節座標轉 SMPL 參數：現有方法候選清單

- 日期：2026-11-15
- 狀態：第一階段（只有調查與清單）。沒有下載模型、沒有跑任何實驗、沒有改現有轉換程式。
- 下一步：Allen 看過後決定實測哪幾個。

## 先講結論

1. **現有版本的 75 公分誤差，看起來不是方法不好，而是關節順序對錯了。**
   `converters/joints_24j_to_smpl.py` 呼叫 `utils/smpl/easymocap_wrapper.py` 時沒有指定骨架種類，預設值是 OpenPose 的 25 點（BODY25）。
   於是程式把輸入的「SMPL 24 關節」（第 0 點是骨盆、第 1 點是左髖…）拿去和模型算出的「BODY25 的 25 點」（第 0 點是鼻子、第 1 點是脖子…）逐點比對，還把補上的第 25 點設成原點、信心值 1。
   這是讀程式得到的判斷，還沒有實際跑過驗證。如果屬實，光是把比對對象改成 SMPL 自己的 24 關節，誤差就應該從 75 公分降到幾公分以內。
2. **這個問題本身有天生的限制：只有關節點，沒辦法完全決定 SMPL 參數。**
   關節點決定得了骨頭「指向哪裡」，決定不了骨頭「繞自己的軸轉了多少」（例如前臂的翻轉、頭的左右轉），也決定不了胖瘦。
   所以各方法在「關節位置」上都能做到很準（公釐到一兩公分），真正分出高下的是「旋轉角度對不對」和「看起來自不自然」。比較標準要照這點設計（見後面草案）。
3. **建議優先實測四個**，都不需要大模型，前兩個連新環境都不用裝：
   - 先在複本上修正現有 EasyMocap 版本的關節對應，當作真正的基準。
   - HybrIK 的解析式解法（直接用公式算旋轉，本機 `external/Hybrik` 已有程式）。
   - VPoser 姿態先驗加最佳化（本機 `external/human_body_prior` 已有官方範例）。
   - Learnable SMPLify（2025 年的神經網路版本，一次前向就算出參數，有公開權重）。

## 名詞說明

- **SMPL 參數**：描述一個人體模型的數字，包含每個關節的旋轉（姿態，72 個數）、體型（10 個數）、整體位移。
- **最佳化擬合**：從一個初始姿態出發，反覆微調參數，直到模型的關節貼近輸入的關節。準但慢，容易卡在奇怪的姿態。
- **姿態先驗**：一個「什麼姿態像真人」的評分器，擬合時用來擋掉不自然的姿態。
- **解析式解法**：不用反覆微調，直接用幾何公式從骨頭方向算出旋轉。極快，但繞骨頭軸的那個轉角（下稱「軸向扭轉」）得另外給。
- **神經網路回歸**：訓練一個網路，輸入關節座標、直接輸出 SMPL 參數。快，但遇到訓練資料沒有的動作（例如四周跳）可能失準。

## 候選方法

實測成本的估法：以「一個人把它裝起來、接上我們的 24 關節輸入、跑完比較用的資料」的人工時間估計，不含等待 Allen 決定的時間。

### A. 建議優先實測

#### A0. 修正關節對應後的現有 EasyMocap 版本（基準）

- 出處：EasyMocap，浙江大學，<https://github.com/zju3dv/EasyMocap>，本機 `external/EasyMocap`。
- 做法：和現在一樣是最佳化擬合，只是把比對對象改成 SMPL 自己的 24 關節。
- 輸入：只要關節點。目前設定不估體型（固定平均體型），骨長不合時會有殘差。
- 授權：EasyMocap 自訂授權，限非商業研究用途（GitHub 標示為非標準授權，細節要讀 LICENSE）。
- GPU：不需要，CPU 可跑，GPU 較快。
- 預估成本：半天以內。在複本上改，不動原檔。
- 備註：repo 裡還有另一支 `joints_24j_to_smpl_smplx_handmade.py`（自己寫的四階段最佳化，會先用體型去配骨長），也應該放進比較，它可能已經比 75 公分好很多。

#### A1. HybrIK 的解析式反向運動學

- 出處：Li 等人，CVPR 2021，<https://github.com/jeffffffli/HybrIK>，本機 `external/Hybrik` 的 `hybrik/models/layers/smpl/lbs.py` 裡有 `batch_inverse_kinematics_transform` 與 `_naive` 兩個函式。
- 做法：沿著骨架由骨盆往外，每個關節用「靜止姿態的骨頭方向」與「目標骨頭方向」直接算出旋轉；軸向扭轉原本由影像網路預測，我們沒有影像，要設成零或另外估。
- 輸入：關節點，加上靜止姿態的骨架（等於要先有體型或骨長）。原版用 29 個點（24 關節再加頭頂、指尖、腳尖），只有 24 點時末端關節的旋轉要另外處理，這點要讀程式確認。
- 授權：程式 MIT。
- GPU：不需要，一次矩陣運算，幾千幀在 CPU 上是秒級。
- 預估成本：一天。主要工作是把函式從影像流程裡單獨抽出來呼叫。
- 預期：關節位置幾乎完全吻合；前臂、頭、腳掌的扭轉會不準。適合當最佳化方法的起點（先用它算出大致姿態，再微調），可以解決最佳化容易卡住的問題。

#### A2. VPoser 姿態先驗加最佳化（human_body_prior 的 IK 引擎）

- 出處：Ghorbani 等人（馬普所），<https://github.com/nghorbani/human_body_prior>，本機 `external/human_body_prior/tutorials/` 有 `ik_example_joints.py`，還有一支專門把動作生成模型輸出的關節轉成 SMPL 的 `mdm_motion2smpl.py`。
- 做法：最佳化擬合，但姿態不是直接調 72 個數，而是在 VPoser 學到的「真人姿態空間」裡調，所以不容易出現扭成麻花的姿態。
- 輸入：只要關節點，可自訂用哪些點；可同時估體型。
- 授權：馬普所授權，限非商業研究用途。
- GPU：不需要，有 GPU 較快。
- 預估成本：一到兩天。本機只有 VPoser 第 1 版的權重（`common_models/vposer_v1_0`），這個引擎多半要第 2 版，需要到官網登入下載（幾十 MB），下載前先問 Allen。
- 備註：repo 裡的 `joints_52j_to_smplh_smplifyx.py` 已經用過類似做法處理 52 關節，可以參考。

#### A3. Learnable SMPLify

- 出處：arXiv 2508.13562（2025 年 8 月），<https://github.com/Charrrrrlie/Learnable-SMPLify>。
- 做法：神經網路回歸，一次前向算出 SMPL 參數。論文宣稱比傳統最佳化快約 200 倍，網格誤差少 5 公釐以上，並在沒看過的資料集（3DPW、RICH）上仍然有效。這些是論文自己的數字，我們沒有驗證過。
- 輸入：關節點。是否需要前一幀結果當起點、接受哪一種關節定義，要讀程式確認。
- 授權：程式 MIT。權重是用 AMASS 資料訓練的，AMASS 本身限非商業用途。
- GPU：推論量小，預估單卡幾 GB 內，CPU 也可能跑得動。
- 預估成本：一到兩天。需要獨立 conda 環境與下載權重（在 Hugging Face，大小未查）。
- 風險：訓練資料是日常動作，高圈數跳躍這類動作可能失準，正好用滑冰片段檢驗。

### B. 值得列入第二輪

#### B1. SMPL-IKS

- 出處：<https://github.com/Z-Z-J/SMPL-IKS>（論文同名，在 IEEE 系列發表）。
- 做法：先由骨架反推體型，再修正姿態，最後用解析式解法算旋轉，軸向扭轉由一個小網路補上。等於是 A1 把「扭轉怎麼辦」補齊的版本。
- 輸入：只要 3D 骨架，這正是它設計的用途；另有 SMPL-H、SMPL-X 版本。
- 授權：GitHub 標示 MIT。
- GPU：小模型，預估不需要或很少。
- 預估成本：兩天。權重放在 Google Drive；repo 只有 8 顆星、2024 年 8 月後沒有更新，文件少，接起來可能要讀原始碼。
- 備註：README 的結果表顯示關節誤差在 1 公釐以下、另一項誤差約 1 到 2.4 公分（應該是網格頂點誤差，要讀論文確認）。

#### B2. joints2smpl（三維版 SMPLify）

- 出處：<https://github.com/wangsen1312/joints2smpl>。MDM、MotionGPT、T2M-GPT 等動作生成專案把關節轉成網格時幾乎都用它（檔名通常是 `simplify_loc2rot.py`），是這個領域實際上的通用做法。
- 做法：逐幀最佳化擬合，用高斯混合模型當姿態先驗，同時估體型，前一幀的結果當下一幀的起點。
- 輸入：只要關節點，原生支援 22 或 24 關節。
- 授權：**repo 沒有授權檔**，法律上等於保留所有權利；裡面的姿態先驗檔來自馬普所的 SMPLify，限非商業用途。自己研究用問題不大，放進會公開的程式或網站前要處理。
- GPU：建議有，逐幀跑、每幀約 150 次迭代，CPU 上會很慢。
- 預估成本：一天。
- 備註：已知缺點是慢、幀間會抖、偶爾手腳翻轉。列入是因為它是別人最常拿來比的對象。

#### B3. DPoser-X（擴散模型姿態先驗）

- 出處：ICCV 2025，<https://github.com/moonbow721/DPoser-X>。
- 做法：和 A2 同一類，但把 VPoser 換成擴散模型當姿態先驗，論文主張對少見姿態比較穩。
- 輸入：關節點（可只給部分關節）。身體模型是 SMPL-X，身體部分 21 個關節的旋轉可以搬到 SMPL，但不是完全等價。
- 授權：程式 MIT；權重用 AMASS 訓練。
- GPU：需要，擴散模型要多步運算，預估 12 GB 內。
- 預估成本：兩到三天。要獨立環境、下載權重，還要自己寫「關節點對齊」的那一段。

### C. 有條件適用

#### C1. SMPLFitter

- 出處：Sárándi 與 Pons-Moll，<https://github.com/isarandi/smplfitter>，MIT，持續維護，可以 `pip` 安裝。
- 做法：把每個身體部位的頂點當剛體，用閉合公式對齊求旋轉，再用線性最小平方求體型，交替幾輪。作者的數字是單張 RTX 3090 每秒約 9,500 次擬合。
- 輸入：**主要輸入是網格頂點**，關節點是輔助。README 明講只給少量點時，旋轉會解不出來。
- 判斷：只有 24 個關節點時不適用。但如果來源是會輸出頂點的偵測器，它是最快最準的選擇，所以留在清單上。

### D. 看過、目前不建議

| 方法 | 不建議的原因 |
| :--- | :--- |
| NIKI、HybrIK-X、IKOL、PLIKS、KAMA | 都是「從影像估人體」的整套系統，裡面的反向運動學模組依賴影像特徵或網格頂點，不能單獨拿關節點來用。NIKI 的 repo 也沒有授權檔。 |
| Fast Neural Inverse Kinematics on Human Body Motions（arXiv 2506.17996，Yoom 公司） | 題目完全吻合，但沒找到公開的程式與權重，訓練資料是公司內部的。 |
| GIK-Net（arXiv 2205.00508）、Skeleton2Mesh（ICCV 2021） | 沒找到可用的公開程式。 |
| MoSh++、SOMA | 輸入是貼在皮膚表面的動捕標記點，不是關節中心，問題設定不同。 |
| Pose-NDF、NRDF | 也是姿態先驗，定位和 A2、B3 重疊；Pose-NDF 自 2023 年後沒有更新，NRDF 的程式位置這次沒有確認到。 |
| minimal-IK-pytorch | 很單純的最佳化實作，沒有姿態先驗，預期不會比 A0 好。 |

搜尋範圍說明：這次用英文關鍵字搜了論文與 GitHub，涵蓋解析式、最佳化、神經網路三類。沒有系統性搜過的地方有：動畫與遊戲業界的工具（例如 Blender 外掛、Unity 的反向運動學套件）、機器人重定向工具、MMHuman3D 內建的三維關節擬合（憑印象它有，沒查證）。所以這份清單不能當作「全部就這些」。

## 比較標準草案

### 資料

1. **有標準答案的資料（主要評分）**：拿 AMASS 的 SMPL 參數，用現有的 `amass_to_smpl.py` 與 `smpl_to_smpl_24j.py` 產生 24 關節座標，再交給各方法轉回 SMPL 參數，和原本的比。
   - 本機 `data/amass/samples_smpl_H_G` 已有樣本。建議再挑 20 到 30 段，刻意包含快速旋轉、跳躍、舞蹈、體操，因為走路這類動作每個方法都會過關。
   - 每段都測三種輸入：原始關節；加上 1 到 2 公分隨機雜訊的關節（模擬偵測器的誤差）；換一個體型算出來的關節（模擬輸入骨長和模型不合）。
2. **Allen 的滑冰動作（沒有標準答案，看穩不穩）**：本機已有偵測結果的五支，`2A+2Lo_0001`、`3A_0001`、`4T_0001`、`4Lz+3T_0001`、`8_front_9_cam2`。輸入用同一個偵測器輸出的 24 關節，各方法都吃同一份。

### 指標

| 指標 | 白話說明 | 適用資料 |
| :--- | :--- | :--- |
| 關節位置誤差（公釐） | 轉回去的模型，24 個關節離輸入的關節平均多遠。分「不做任何對齊」與「把骨盆對齊後」兩種。 | 兩種資料都算 |
| 姿態旋轉誤差（度） | 每個關節的旋轉和標準答案差幾度。四肢末端（手腕、手掌、腳掌、頭）分開報，因為那些關節的扭轉本來就難以決定。 | 只有 AMASS |
| 網格頂點誤差（公釐） | 整個身體表面平均差多遠，可以反映扭轉與體型有沒有對。 | 只有 AMASS |
| 抖動 | 相鄰幀之間姿態變化的劇烈程度，和輸入本身的抖動相比。 | 兩種資料都算 |
| 空中圈數保留 | 由骨盆朝向累計的旋轉圈數，和直接由輸入關節算出的圈數相比。這是針對滑冰加的。 | 滑冰 |
| 速度 | 每 1,000 幀要幾秒，註明用 CPU 還是哪張 GPU。 | 兩種資料都算 |
| 人工檢查 | 每支滑冰片段渲染成網格影片，看有沒有手腳翻轉、身體扭成麻花。 | 滑冰 |

### 基準與過關線

- 要超越的基準：現有 `joints_24j_to_smpl.py` 的 75 公分。
- 但 75 公分很可能只是關節對應的錯誤，所以建議同時把 A0（修正對應後的版本）和 `joints_24j_to_smpl_smplx_handmade.py` 也當基準。新方法要贏過這兩個才有採用的意義。
- 建議的過關線（待 Allen 確認）：AMASS 原始關節輸入下，關節位置誤差 2 公分以內；滑冰片段沒有肉眼可見的翻轉，圈數不掉。

### 執行方式

- 第一輪只測 A0、A1、A2 與手寫版本：都用現有的 `skeleton_env`，不裝新東西，CPU 就能跑。A2 可能要下載 VPoser 第 2 版權重，下載前先問。
- 第二輪才測 A3 與 B 類：每個方法一個獨立 conda 環境，用 `python -m pip` 安裝。要用 GPU 時，在 `neverdie` 既有的預先同意範圍內（單卡、12 GB 內、2 小時內，先跑 `~/hq/bin/gpufree`），超出就另外問 Allen。

## 授權提醒

- **SMPL 模型本身限非商業用途。** 不管用哪個方法，只要輸出的是 SMPL 參數、要放到網站上，而網站有商業性質，就需要向 Meshcapade 另外取得商業授權。這一點和選哪個方法無關。
- 限非商業研究用途的元件：EasyMocap、VPoser 與 human_body_prior、SMPLify-X、SMPLify 的高斯混合姿態先驗（joints2smpl 用到）。
- 程式是 MIT 的：HybrIK、SMPLFitter、Learnable SMPLify、SMPL-IKS、DPoser-X。但後三者的權重是用 AMASS 訓練的，AMASS 限非商業用途，權重能不能商用要另外確認。
- 沒有授權檔的：joints2smpl、NIKI。

## 還沒確認的事

- 「75 公分是關節順序對錯」是讀程式的判斷，沒有實際跑過。
- HybrIK 的解析式函式在只有 24 點、沒有影像網路的情況下要怎麼呼叫。
- Learnable SMPLify 接受的關節定義、是否逐幀獨立、權重大小。
- human_body_prior 的 IK 引擎需要哪一版 VPoser。
- SMPL-IKS 的 README 結果表各欄的確切意義。

## 來源

- HybrIK：<https://arxiv.org/abs/2011.14672>、<https://github.com/jeffffffli/HybrIK>
- NIKI：<https://openaccess.thecvf.com/content/CVPR2023/papers/Li_NIKI_Neural_Inverse_Kinematics_With_Invertible_Neural_Networks_for_3D_CVPR_2023_paper.pdf>
- Learnable SMPLify：<https://arxiv.org/html/2508.13562v1>、<https://github.com/Charrrrrlie/Learnable-SMPLify>
- SMPL-IKS：<https://github.com/Z-Z-J/SMPL-IKS>、<https://ieee-dataport.org/node/13979>
- joints2smpl：<https://github.com/wangsen1312/joints2smpl>
- human_body_prior（VPoser）：<https://github.com/nghorbani/human_body_prior>
- SMPLFitter：<https://github.com/isarandi/smplfitter>、<https://smplfitter.readthedocs.io/en/latest/>
- DPoser-X：<https://arxiv.org/abs/2508.00599>、<https://github.com/moonbow721/DPoser-X>
- Fast Neural Inverse Kinematics on Human Body Motions：<https://arxiv.org/abs/2506.17996>
- GIK-Net：<https://arxiv.org/abs/2205.00508>；KAMA：<https://arxiv.org/abs/2104.13502>；IKOL：<https://arxiv.org/abs/2302.01058>；PLIKS：<https://arxiv.org/abs/2211.11734>
- Skeleton2Mesh：<https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Skeleton2Mesh_Kinematics_Prior_Injected_Unsupervised_Human_Mesh_Recovery_ICCV_2021_paper.pdf>
- Pose-NDF：<https://github.com/garvita-tiwari/PoseNDF>；NRDF：<https://virtualhumans.mpi-inf.mpg.de/nrdf>
- EasyMocap：<https://github.com/zju3dv/EasyMocap>
- SMPL 授權：<https://smpl.is.tue.mpg.de/license.html>
