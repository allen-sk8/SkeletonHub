#!/bin/bash
# 10 whole-match FineFS videos: Stage A (tracking, reidtrack env) then Stage B (gvhmr_fast, skeleton_env). One GPU.
# usage: CUDA_VISIBLE_DEVICES=<free card> bash scripts/run_finefs10.sh
cd /home/allen/SkeletonHub
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
IDS=36,261,819,739,980,652,187,513,521,456
OUT=outputs/finefs_full_test
PY=/home/allen/miniconda3/envs
echo "=== stage A tracking $(date)"
$PY/reidtrack/bin/python scripts/finefs_full_track.py --ids $IDS --out $OUT --tracker botsort 2>&1 | tee -a $OUT/stageA.log
echo "=== stage B gvhmr_fast $(date)"
$PY/skeleton_env/bin/python scripts/finefs_full_fast.py --ids $IDS --out $OUT --device cuda:0 2>&1 | tee -a $OUT/stageB.log
echo "=== ALL DONE $(date)" | tee -a $OUT/stageB.log
