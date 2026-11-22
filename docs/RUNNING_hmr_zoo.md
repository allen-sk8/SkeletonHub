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
