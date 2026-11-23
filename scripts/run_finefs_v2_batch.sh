#!/bin/bash
# FineFS v2 batch (one GPU, two lanes on the same card + watchdog). Resumable: finished stages are skipped.
#   GPU=<card> IDS=<comma list> R=outputs/finefs_v2_batch100 bash scripts/run_finefs_v2_batch.sh {track|gvhmr|watchdog}
# track   : RF-DETR detection + BoT-SORT with in-tracker Re-ID + online lock for all ids -> $R/_track/<id>
# gvhmr   : as tracking results appear: offline lock -> $R/_lock/<id>, 0_video + 1_tracking, then in chunks of 5:
#           gvhmr_fast (half resolution only; saves 2D) -> 2_pose2d, 3_gvhmr; flags, preview, per-video checks
# watchdog: every 60 s stops everything (touch $R/STOP, kill this batch's python processes) if another process appears
#           on the card, our memory > MAX_MIB, or wall time > MAX_H hours.
cd /home/allen/SkeletonHub
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ENVS=/home/allen/miniconda3/envs
R=${R:-outputs/finefs_v2_batch100}
MAX_MIB=${MAX_MIB:-14000}; MAX_H=${MAX_H:-24}; CHUNK=${CHUNK:-5}
mkdir -p $R
log() { echo "[$(date '+%F %T')] $*" | tee -a $R/batch.log; }

case "$1" in
track)
  log "track lane start (gpu $GPU)"
  $ENVS/reidtrack/bin/python scripts/finefs_full_track.py --ids $IDS --out $R/_track --tracker botsort-reid --gpu-mem-gb 6 2>&1 | tee -a $R/track.log
  log "track lane DONE"; touch $R/_track/ALL_DONE ;;
gvhmr)
  log "gvhmr lane start (gpu $GPU)"
  ready=()
  run_chunk() {
    [ ${#ready[@]} -eq 0 ] && return
    local ch=$(IFS=,; echo "${ready[*]}")
    log "gvhmr chunk $ch"
    $ENVS/skeleton_env/bin/python scripts/finefs_v2_gvhmr_dual.py --variants half --ids $ch --boxes-root $R/1_tracking --out $R 2>&1 | tee -a $R/gvhmr.log
    $ENVS/reidtrack/bin/python scripts/finefs_v2_export.py stage4 --root $R --no-summary --ids $ch 2>&1 | grep -v '^\[check\]' | tee -a $R/export.log
    ready=()
  }
  for id in ${IDS//,/ }; do
    while [ ! -f $R/_track/$id/track_stats.json ]; do
      [ -f $R/STOP ] && { log "gvhmr lane stopped"; exit 1; }
      if [ -f $R/_track/ALL_DONE ]; then log "WARNING $id has no tracking result"; continue 2; fi
      run_chunk; sleep 30
    done
    [ -f $R/STOP ] && { log "gvhmr lane stopped"; exit 1; }
    if grep -q '"no_person": true' $R/_track/$id/track_stats.json; then log "$id: no person locked, skipped"; continue; fi
    if [ ! -f $R/1_tracking/$id/boxes.json ]; then
      $ENVS/reidtrack/bin/python preprocessors/track_person_offline.py --video /home/allen/datasets/FineFS/video/$id.mp4 \
        --track-dir $R/_track/$id --out-dir $R/_lock/$id 2>&1 | tail -1 | sed "s/^/[lock] $id /" | tee -a $R/batch.log
      $ENVS/reidtrack/bin/python scripts/finefs_v2_export.py stage1 --root $R --ids $id --boxes-from $R/_lock --tracks-from $R/_track >> $R/export.log 2>&1
    fi
    [ -f $R/3_gvhmr/$id/gvhmr_fast.npz ] && [ -f $R/5_preview/$id.mp4 ] && continue
    ready+=($id)
    [ ${#ready[@]} -ge $CHUNK ] && run_chunk
  done
  run_chunk
  $ENVS/reidtrack/bin/python scripts/finefs_v2_export.py stage4 --root $R --ids $IDS 2>&1 | grep -v '^\[check\]' | tee -a $R/export.log
  log "gvhmr lane DONE"; touch $R/GVHMR_DONE ;;
watchdog)
  t0=$(date +%s)
  log "watchdog start (gpu $GPU, max ${MAX_MIB} MiB, ${MAX_H} h)"
  while [ ! -f $R/GVHMR_DONE ] && [ ! -f $R/STOP ]; do
    reason=""
    tot=0
    while IFS=, read pid mem; do
      pid=${pid// /}; mem=${mem//[^0-9]/}; [ -z "$pid" ] && continue; mem=${mem:-0}
      args=$(ps -o args= -p $pid 2>/dev/null)
      [ -z "$args" ] && continue                      # process already gone (or not visible): ignore
      if [[ "$args" != *finefs_v2_batch100* && "$args" != *"$R"* ]]; then
        sleep 10; nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | grep -qw "$pid" && \
          reason="foreign process on card: pid $pid $(ps -o user= -p $pid) ${args:0:80}"
      fi
      tot=$((tot + mem))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $GPU)
    [ $tot -gt $MAX_MIB ] && reason="our memory ${tot} MiB > ${MAX_MIB}"
    [ $(( $(date +%s) - t0 )) -gt $((MAX_H * 3600)) ] && reason="wall time > ${MAX_H} h"
    if [ -n "$reason" ]; then
      log "STOP: $reason"; touch $R/STOP
      pkill -f "python.*finefs_v2_batch100"; sleep 5; pkill -9 -f "python.*finefs_v2_batch100"
      break
    fi
    echo "$(date '+%T') ${tot}MiB" > $R/watchdog.last
    sleep 60
  done
  log "watchdog end" ;;
esac
