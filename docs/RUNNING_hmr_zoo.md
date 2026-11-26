# hmr-zoo RUNNING log (GPU runs)

Branch `feat/hmr-zoo`, worktree `/home/allen/SkeletonHub.hmr-zoo`. Machine `neverdie`. One line per start/end.
- 2026-11-22 23:03 start: setup, no GPU yet
- 2026-11-22 23:04 start neverdie GPU0 (TITAN RTX, empty per gpufree): phase-1 7 models on FineFS 3A_3A_0133 (125 frames), est. 15 min, <8 GB each. tmux skhub-hmrzoo
- 2026-11-22 23:04-23:13 neverdie GPU0 phase-1 first attempt (worktree, now removed): hybrik/gvhmr failed (numpy from ~/.local shadowing), wham OK (125f, 4.2 fps, 5.32 GB); stopped by foreign-process check (false alarm, lingering pid of own process). GPU 0 released (9 MiB).
- 2026-11-22 23:13 start neverdie GPU0 again (gpufree OK, 5 other empty cards): phase-1 remaining 6 models on FineFS 3A_3A_0133, est 15 min, <8 GB. tmux skhub-hmrzoo
- 2026-11-22 23:17 GPU0 now used by a colleague (jason) -> stopped, not using GPU0. Moving to GPU1 (TITAN RTX, gpufree OK).
- 23:33 GPU1: hybrik,gvhmr,instanthmr,nlf,sam3db done for 3A; onlinehmr hit transient 'permission denied' on env python (NFS); rerunning
- 23:35 GPU1: onlinehmr done; phase-1 7 models on 3A OK. Now tracker validation (RF-DETR+BoT-SORT, <=4GB cap) on GPU1
- 23:38 tracker validation done (boxes identical to finefs, 0.03 px). GPU1: preparing other clips (tracking only, <=4GB)
- 23:46 start neverdie GPU1 (empty per gpufree; 5 other cards empty): phase-2 all 8 models x 14 clips (~1500 frames), est 60-80 min, each model <8 GB. tmux skhub-hmrzoo
- 00:36 GPU1 phase-2 model inference done (8 models x 12 clips, ~55 min); GPU1 memory back to 9 MiB. Render (CPU) continuing.
- 00:37 start neverdie GPU0 (gpufree ok): FlowHMR probe on 3A clip (vggt_interval 10, hard cap 23.5 GB), est <15 min. tmux skhub-flow
- 00:43 FlowHMR probe on GPU0: OOM at 23.5 GB cap (VGGT-Omega attention, vggt_interval 10) on TITAN RTX 24GB; stopped, GPU0 back to 9 MiB. Not retried (needs A6000/>24GB -> Allen to decide).
- 00:44 all done. No GPU in use by this task (GPU0/1 at 9 MiB), tmux skhub-hmrzoo and skhub-flow closed.
- 2026-11-23 01:15 start neverdie GPU0 (gpufree ok, 5 other empty): gvhmr_fast development + 2-clip check (3A_3A_0133, 4Lz_4Lz_0018), est 15 min, <8 GB. tmux skhub-finefs10 (dev runs use this session name)
- 2026-11-23 01:22 gvhmr_fast 2-clip check done (GPU0 released, 9 MiB). start neverdie GPU0 (gpufree ok, 5 other empty): FineFS 10 whole-match videos, track (<=6 GB) then gvhmr_fast (~3 GB), est ~1 h, stop if >3 h or >20 GB or foreign process. tmux skhub-finefs10
- 2026-11-23 01:30 stopped FineFS-10 run after video 36 (lock rule picked a spectator; fixing rule, rerun). GPU0 released.
- 2026-11-23 01:40 restart FineFS-10 on neverdie GPU0 (gpufree ok): tracking (RF-DETR + BoT-SORT without in-tracker Re-ID, lock step uses OSNet Re-ID) then gvhmr_fast; est 1 h; tmux skhub-finefs10
- 2026-11-23 02:31 start neverdie GPU0 (gpufree ok): stage B gvhmr_fast on 10 whole FineFS videos (stage A tracking done, lock rule fixed), est 45 min, <4 GB expected. tmux skhub-finefs10
- 2026-11-23 02:38 restart stage B (SimpleVO made failure-tolerant) neverdie GPU0 (gpufree ok), est 45 min. tmux skhub-finefs10
- 2026-11-23 03:01 FineFS-10 stage B done, GPU0 released (9 MiB), tmux closed. Next: 5s-clip comparison (GPU0, est 10 min).
- 2026-11-23 03:02 start neverdie GPU0 (gpufree ok): 5s-clip comparison (original GVHMR + precise vs whole-video gvhmr_fast, 41 clips), est 15 min, <8 GB. tmux skhub-finefs10
- 2026-11-23 03:11 neverdie GPU0 (same card as my clip-compare run): rerun gvhmr_fast for videos 456, 652 after lock-rule tweak, est 8 min, <8 GB. tmux skhub-finefs10b
- 2026-11-23 03:43 all GPU work of the FineFS-10 test finished; GPU0 back to 9 MiB, tmux skhub-finefs10 / skhub-finefs10b closed.
- 2026-11-23 22:32 start neverdie GPU0 (gpufree ok, 3 other empty cards left): FineFS v2 test exp2 = botsort-reid tracking on the 10 whole videos (cached detections), est 45 min, <=6 GB. tmux skhub-v2test
- 2026-11-23 23:22 exp2 done (GPU0 back to idle). start neverdie GPU0 (gpufree ok, 3 other empty cards left): FineFS v2 test exp3 = gvhmr_fast with half- and full-resolution crops on the 10 whole videos, est 60-75 min, <8 GB; stop if >2.5 h. tmux skhub-v2test
- 2026-11-24 00:05 exp3 done, GPU0 back to 9 MiB, tmux skhub-v2test closed. Rest of the v2 test (flags, previews, checks) is CPU only.
- 2026-11-24 00:39 start neverdie GPU0 (gpufree ok, 3 other empty cards left; Allen approved in person, parallel ~12 h): FineFS v2 batch of 100 random videos (5.6 h video), lanes track (botsort-reid) + gvhmr_fast half-res on the same card, est ~11 GB, ~12 h; watchdog stops at foreign process / >14 GB / >24 h. tmux skhub-v2batch
- 2026-11-24 01:00 FineFS v2 batch: tracking process was using ~16 of 48 CPU cores (31% system time, thread contention); pinned live with taskset: tracking lane cores 0-7, gvhmr lane cores 8-13 (machine idle went 46% -> 70%).
- 2026-11-24 02:04 FineFS v2 batch STOPPED by watchdog at 02:02: an ollama server (user ollama) started on GPU0 using 14.6 GB (logged as 'memory > 14000' because the watchdog counted it as ours; fixed). Done: 10 of 100 videos complete, 11 tracked. All our processes ended, tmux skhub-v2batch closed. gpufree now reports 0 empty cards on neverdie, so not resumed; needs Allen.
- 2026-11-24 09:52 start neverdie GPU5 (RTX 3090; gpufree ok, 4 other empty cards left; Allen approved in person in this session, long-term use of this card, FOREIGN_STOP=0): FineFS v2 FULL batch, 1,147 videos left of 1,167 (BoT-SORT Re-ID + offline lock, half-res gvhmr_fast, previews), est ~8 days, ~11 GB; stops at our memory >14 GB, free disk <300 GB, 3 failures in a row, >16 days; moves to another empty card on out-of-memory caused by another user. tmux skhub-v2full. Progress: outputs/finefs_v2_full/PROGRESS.md
- 2026-11-24 12:34 FineFS v2 full batch PAUSED by Allen (to settle jump height and horizontal travel first). 41 of 1167 done, 0 failed. GPU5 released (18 MiB), tmux skhub-v2full closed. Partial preview 24.mp4 (interrupted) removed; 3D/2D/preview files are now written as .part and renamed when complete. Resume: GPU=<card> bash scripts/run_finefs_v2_full.sh
- 2026-11-24 14:46 RESUME neverdie GPU5 (RTX 3090; gpufree: 6 empty cards, 5 left after this; Allen approved in person: same settings as before): FineFS v2 full batch stages 1-5, 1,127 videos left, est ~6 days. Same tmux skhub-v2full also runs stage 6 (CPU only, 2 workers on cores 16-27, ~3 min per video, keeps up with the GPU). Progress: outputs/finefs_v2_full/PROGRESS.md
- 2026-11-24 15:05 stage 6 workers had opened a small CUDA context (258 MiB each) on GPU2 (A6000) because CUDA_VISIBLE_DEVICES was not set; noticed after ~18 min, killed, restarted without GPU access; the script now hides all GPUs itself. GPU2 back to 18 MiB.
- 2026-11-25 15:08 FineFS v2 full batch PAUSED (Allen, via pm: card 5 needed for HW3). 268 of 1167 done. GPU5 released (18 MiB). Video 260 was interrupted mid-run (exit 143), retried on resume.
- 2026-11-26 09:42 RESUME neverdie GPU5 (RTX 3090; Allen approved in person on 11-26, multi-day single card; only card 4 left fully empty, below the usual 2, so no extra cards). Started by pm, monitored by the SkeletonHub agent. 899 videos left, est ~4 days. tmux skhub-v2full (stages 1-5), skhub-v2full-s6w1 / -s6w2 (stage 6, CPU only). Progress: outputs/finefs_v2_full/PROGRESS.md

## FineFS v2 全批：交接給 pm 的每日檢查（2026-11-26 起）

Allen 決定關掉 SkeletonHub 的對話，每日檢查改由 pm 負責。重跑本身在三個 tmux 裡繼續，不受影響：
`skhub-v2full`（第 1 到 5 階段，用 neverdie 5 號卡）、`skhub-v2full-s6w1` 和 `skhub-v2full-s6w2`（第 6 階段，只用 CPU）。
2026-11-26 09:42 續跑時還剩 899 支，估計約 4 天跑完。

### 1. 每天跑這一段（只讀檔，幾秒鐘，不碰 GPU）

```bash
cd /home/allen/SkeletonHub/outputs/finefs_v2_full
date; head -9 PROGRESS.md
echo "-- 監督腳本狀態:"; cat supervisor.last
echo "-- 看門程式最後一次記錄（每分鐘寫一次）:"; cat watchdog.last
echo "-- 記錄檔最後 5 行:"; tail -n 5 batch.log | cut -c1-160
echo "-- 停止或完成的標記:"; ls STOP STOP6 ALL_DONE 2>/dev/null
echo "-- 失敗清單:"; ls _failed _failed6
echo "-- tmux:"; tmux ls | grep skhub-v2full
echo "-- 5 號卡上的程序:"; nvidia-smi -i 5 --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read p m; do echo "$p $m $(ps -o user= -p $p)"; done
echo "-- 硬碟:"; df -h /home | tail -1
```

`PROGRESS.md` 每 10 分鐘自動更新一次，裡面有完成數、失敗數、剩幾支、估計剩多久，也有第 6 階段的完成數和失敗數。
估計時間是用最近一天的平均速度算的，剛續跑時會偏高（暫停的時間也被算進去），正常大約每小時 9 支。

### 2. 怎樣算異常

- **自己停了**：`supervisor.last` 以 `stopped:` 開頭，或者 `STOP` 存在、而 `batch.log` 裡出現 `STOP for good`。
  停的原因會寫在那一行：連續 3 支失敗、記憶體超過 14000 MiB、硬碟剩不到 300 GB、跑超過 384 小時，或連續 3 次在沒人用的卡上記憶體不足。
  注意：如果 `STOP` 的內容是 `oom` 開頭，代表有同事在用這張卡、我們被擠到記憶體不足。這時腳本會自己換到另一張空卡繼續跑；如果沒有空卡，會每 30 分鐘再查一次。這種情況不是異常，但要回報 pm，因為它會多用一張卡。
- **卡住**：`watchdog.last` 的時間比現在晚超過 5 分鐘，或 `PROGRESS.md` 的更新時間比現在晚超過 30 分鐘，或 `batch.log` 超過 1 小時沒有新的一行，或「Finished」的數字一整天沒有增加（正常一天約 200 支）。
- **失敗**：`_failed/` 或 `_failed6/` 裡出現新檔案。單支失敗會被記下並跳過，批次照樣繼續。偶爾一兩支不用處理，但一天超過 5 支就算異常。
- **記憶體**：5 號卡上我們的程序加起來超過約 12 GB（平常 1 到 9 GB）。
- **硬碟**：`/home` 剩不到約 500 GB。腳本到 300 GB 會自己停。
- **同事用到同一張卡**：5 號卡上出現不是 allen 的程序。依照 Allen 11-24 的設定（`FOREIGN_STOP=0`），批次不會因此停下，只會在 `batch.log` 記一行 `note (not stopping ...)`；真的擠到記憶體不足時才會自己換卡。看到這種情況要記下來回報 Allen，由他決定要不要讓出這張卡。
- **tmux 少了一個**：三個 session 裡有一個不見，而且還沒出現 `ALL_DONE`。

### 3. 異常時誰處理

pm 開一個新的 SkeletonHub 對話來處理，把這一節和看到的檢查輸出交給它。pm 自己不重啟、不刪檔、不改設定。
要重啟或換卡（多佔一張卡），仍然要 Allen 本人同意。
新對話續跑的方法：在 tmux `skhub-v2full` 執行 `GPU=5 bash scripts/run_finefs_v2_full.sh`，已完成的會自動跳過。
如果之前是被手動中斷的（失敗原因是 `exit 143`），先把那支在 `_failed/` 裡的記錄移掉再續跑，否則它會被跳過。
第 6 階段的兩個 CPU 程式如果已經結束，要另外重開。指令照 `scripts/finefs_v2_stage6.py` 開頭的說明（`--watch --work outputs/finefs_v2_full --worker 1` 和 `--worker 2`），用 `taskset -c 16-21` 和 `22-27` 綁定 CPU 核心。

### 4. 跑完的收尾

監督腳本結束時會出現 `ALL_DONE`，`supervisor.last` 會變成 `finished ...`，`batch.log` 最後會有 `ALL DONE`。
第 6 階段的兩個 CPU 程式會在做完剩下的影片後自己結束，記錄檔最後一行是 `nothing left and the GPU batch is not running, ending`。
1. 等兩個第 6 階段程式都結束，`PROGRESS.md` 的第 6 階段完成數加失敗數等於總數。
2. 確認 5 號卡上沒有 allen 的程序（上面那條 `nvidia-smi -i 5` 指令沒有 allen）。
3. 關掉三個 tmux：`tmux kill-session -t skhub-v2full; tmux kill-session -t skhub-v2full-s6w1; tmux kill-session -t skhub-v2full-s6w2`
4. 檢查結果：
   - `PROGRESS.md` 的 Finished 加 Failed 加 No skater found 等於 1167。
   - 列出 `_failed/` 和 `_failed6/` 裡每一支失敗的原因。
   - 看 `export.log` 和 `outputs/finefs_v2_full/check.json`（跑完時自動產生的整批檢查）有沒有警告。
   - 抽看幾支 `/home/allen/datasets/FineFS/5_preview/<編號>.mp4`，以及 `outputs/finefs_v2_stage6_previews/` 裡的第 6 階段預覽。
5. 更新檔案：在這個檔加一行完成紀錄（時間、完成數、失敗數、5 號卡已空出），寫進 `~/hq/FROM_AGENTS.md`，然後 commit（只加自己改的檔）。
6. 結果都在 `/home/allen/datasets/FineFS/` 的 `1_tracking`、`2_pose2d`、`3_gvhmr`、`4_flags`、`5_preview`、`6_world`。
   重播和溶接畫面要不要標記，仍等 Allen 決定。
