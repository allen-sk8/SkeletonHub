#!/bin/bash
cd /home/allen/SkeletonHub
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
IDS=36,261,819,739,980,652,187,513,521,456
OUT=outputs/finefs_full_test
/home/allen/miniconda3/envs/skeleton_env/bin/python scripts/finefs_full_fast.py --ids $IDS --out $OUT --device cuda:0 2>&1 | tee -a $OUT/stageB.log
echo "=== ALL DONE $(date)" | tee -a $OUT/stageB.log
