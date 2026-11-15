#!/bin/bash
# 渲染 FineFS 六筆跳躍的並排影片（純 CPU，每筆一個行程）
export CUDA_VISIBLE_DEVICES=""
PY=/home/allen/miniconda3/envs/skeleton_env/bin/python
cd "$(dirname "$0")"
for c in 3A_0018 4T_0018 4S_0044 3Lz+3T_0009 2A_0003 3Lo_0033; do
  $PY render_compare.py finefs --clips "$c" &
done
wait
echo DONE
