# pose2d_bench RUNNING log (GPU runs)

工作目錄：`/home/allen/SkeletonHub/experiments/pose2d_bench/`，輸出：`/home/allen/SkeletonHub/outputs/pose2d_zoo/`，機器：`neverdie`。
規則：開跑前一律執行 `/home/allen/hq/bin/gpufree`；一次只用 1 張完全空的卡；指令前加 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<id>`；單卡預估顯存 <= 12 GB、單段 <= 2 小時；跑完確認顯存釋放並補「結束」。

- 2026-11-23 15:02 stage 0: 模型選定與計畫寫入 `/home/allen/hq/reviews/2026-11-23_pose2d/STATUS.md`（未動 GPU）
- 2026-11-23 16:08 Pilot (4 clips: `3A_3A_0133`, `v25_S_14s`, `101b_0001`, `pitch_11`) 開始：GPU `0` (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`), tmux session `pose2d_pilot`, 依序執行 ViTPose-H+gvhmr_shared / RTMW-x / Sapiens2-0.8B / SDPose-Body / GVHMR / eval_and_render（進行中）
- 2026-11-23 17:06 Pilot 結束：4 片段全部完成（2D/3D 峰值顯存 ViTPose-H 5.78 GB / Sapiens2-0.8B 5.54 GB / SDPose-Body 4.28 GB / RTMW-x 1.14 GB / GVHMR 0.73 GB）；已確認 GPU 0 顯存釋放回 `9 MiB`，`pose2d_pilot` tmux session 已結束。
- 2026-11-23 17:09 Full (全部 12 clips, `clips.txt`) 開始：GPU `0` (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`), tmux session `pose2d_full`, 透過 `--skip-existing` 沿用 4 個 Pilot 快取並補跑其餘 8 個片段與全 12 支並排視覺化（進行中）
- 2026-11-23 21:57 pipeline_table: GPU 0 (neverdie, CUDA_DEVICE_ORDER=PCI_BUS_ID), 12 支片段跑 BoT-SORT 不含外觀再識別追蹤框（RF-DETR 偵測，約 2 GB），預估 15 分鐘內；3D 的 OnlineHMR 沿用 hmr_zoo 現成結果不重跑（進行中）
- 22:04 pipeline_table 追蹤 C 結束，GPU 0 顯存已確認釋放（9 MiB），tmux pipe_trackC 已結束。
