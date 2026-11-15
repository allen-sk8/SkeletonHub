#!/bin/bash
# 渲染第一輪所有並排影片（純 CPU，每個片段一個行程）
export CUDA_VISIBLE_DEVICES=""
PY=/home/allen/miniconda3/envs/skeleton_env/bin/python
cd "$(dirname "$0")"
for c in 2A+2Lo_0001_gvhmr 2A+2Lo_0001_hybrik 8_front_9_cam2_hybrik; do
  $PY render_compare.py skating --clips "$c" &
done
for c in walking_01 running_on_the_spot light_hopping_loose jumping_jacks punch_kick special_move; do
  $PY render_compare.py amass --clips "$c" &
done
wait
echo DONE
