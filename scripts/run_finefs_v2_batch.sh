#!/bin/bash
# FineFS v2 batch (one GPU, two lanes on the same card + watchdog). Resumable: finished stages are skipped.
#   GPU=<card> IDS=<comma list> [W=...] [R=...] bash scripts/run_finefs_v2_batch.sh {track|gvhmr|watchdog}
# R = dataset root that receives the numbered stage folders 1_tracking .. 5_preview (default /home/allen/datasets/FineFS,
#     next to the original video/ annotation/ ... folders; nothing else is written there except stages_info.txt).
# W = work folder inside SkeletonHub for intermediate results, logs, STOP flag and the batch check.json
#     (default outputs/finefs_v2_batch100).
# track   : RF-DETR detection + BoT-SORT with in-tracker Re-ID + online lock for all ids -> $W/_track/<id>
# gvhmr   : as tracking results appear: offline lock -> $W/_lock/<id>, 1_tracking, then in chunks of 5:
#           gvhmr_fast (half resolution only; saves 2D) -> 2_pose2d, 3_gvhmr; flags, preview, per-video checks
# watchdog: every 60 s stops everything (touch $W/STOP, kill this batch's python processes) if another process appears
#           on the card (only with FOREIGN_STOP=1), our memory > MAX_MIB, free disk < MIN_FREE_GB, wall time > MAX_H hours,
#           or a lane wrote a reason into $W/STOP (out of memory: "oom ...", repeated failures: "failures ...").
# Per-video failures are recorded in $W/_failed/<id>.<stage>.txt and skipped (scripts/finefs_failguard.py).
# For the whole dataset use scripts/run_finefs_v2_full.sh, which runs these three lanes and moves to another card on
# out-of-memory caused by another user's process.
cd /home/allen/SkeletonHub
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ENVS=/home/allen/miniconda3/envs
W=${W:-outputs/finefs_v2_batch100}
R=${R:-/home/allen/datasets/FineFS}
# our python processes all carry $W or $R/1_tracking in their arguments (used by the watchdog to tell ours from others')
OURS="python.*(${W}|${R}/1_tracking)"
MAX_MIB=${MAX_MIB:-14000}; MAX_H=${MAX_H:-24}; CHUNK=${CHUNK:-5}
FOREIGN_STOP=${FOREIGN_STOP:-1}; MIN_FREE_GB=${MIN_FREE_GB:-300}
TRACK_CPUS=${TRACK_CPUS:-0-7}; GVHMR_CPUS=${GVHMR_CPUS:-8-13}
F=$W/_failed
mkdir -p $W
log() { echo "[$(date '+%F %T')] $*" | tee -a $W/batch.log; }
# python exit code 3 = CUDA out of memory, 4 = several videos in a row failed (finefs_failguard.py)
on_rc() { case $1 in 3) echo "oom $2" > $W/STOP; log "STOP: out of memory in $2";; 4) echo "failures $2" > $W/STOP; log "STOP: repeated failures in $2";; esac; }
failed() { ls $F/$1.*.txt >/dev/null 2>&1; }

case "$1" in
track)
  log "track lane start (gpu $GPU)"
  taskset -c $TRACK_CPUS $ENVS/reidtrack/bin/python scripts/finefs_full_track.py --ids $IDS --out $W/_track --tracker botsort-reid \
    --gpu-mem-gb 6 --fail-dir $F 2>&1 | tee -a $W/track.log
  on_rc ${PIPESTATUS[0]} track
  if [ -f $W/STOP ]; then log "track lane stopped"; else log "track lane DONE"; touch $W/_track/ALL_DONE; fi ;;
gvhmr)
  log "gvhmr lane start (gpu $GPU)"
  ready=()
  run_chunk() {
    [ ${#ready[@]} -eq 0 ] && return
    local ch=$(IFS=,; echo "${ready[*]}")
    log "gvhmr chunk $ch"
    taskset -c $GVHMR_CPUS $ENVS/skeleton_env/bin/python scripts/finefs_v2_gvhmr_dual.py --variants half --ids $ch --boxes-root $R/1_tracking \
      --out $R --fail-dir $F 2>&1 | tee -a $W/gvhmr.log
    on_rc ${PIPESTATUS[0]} gvhmr
    [ -f $W/STOP ] && { ready=(); return; }
    taskset -c $GVHMR_CPUS $ENVS/reidtrack/bin/python scripts/finefs_v2_export.py stage4 --root $R --no-summary --ids $ch --fail-dir $F 2>&1 \
      | grep -v '^\[check\]' | tee -a $W/export.log
    on_rc ${PIPESTATUS[0]} export
    ready=()
  }
  for id in ${IDS//,/ }; do
    while [ ! -f $W/_track/$id/track_stats.json ]; do
      [ -f $W/STOP ] && { log "gvhmr lane stopped"; exit 1; }
      if [ -f $W/_track/ALL_DONE ]; then log "WARNING $id has no tracking result"; continue 2; fi
      failed $id && { log "$id: failed in an earlier stage, skipped"; continue 2; }
      run_chunk; sleep 30
    done
    [ -f $W/STOP ] && { log "gvhmr lane stopped"; exit 1; }
    if grep -q '"no_person": true' $W/_track/$id/track_stats.json; then log "$id: no person locked, skipped"; continue; fi
    failed $id && continue
    if [ ! -f $R/1_tracking/$id/boxes.json ]; then
      taskset -c $GVHMR_CPUS $ENVS/reidtrack/bin/python preprocessors/track_person_offline.py --video /home/allen/datasets/FineFS/video/$id.mp4 \
        --track-dir $W/_track/$id --out-dir $W/_lock/$id > $W/_lock_$id.log 2>&1
      rc=$?; tail -1 $W/_lock_$id.log | sed "s/^/[lock] $id /" | tee -a $W/batch.log
      if [ $rc -ne 0 ]; then mkdir -p $F; { echo "$(date '+%F %T') error (exit $rc)"; cat $W/_lock_$id.log; } > $F/$id.lock.txt
        log "$id: offline lock failed, skipped"; rm -f $W/_lock_$id.log; continue; fi
      rm -f $W/_lock_$id.log
      $ENVS/reidtrack/bin/python scripts/finefs_v2_export.py stage1 --root $R --ids $id --boxes-from $W/_lock --tracks-from $W/_track >> $W/export.log 2>&1
      if [ $? -ne 0 ] || [ ! -f $R/1_tracking/$id/boxes.json ]; then mkdir -p $F; echo "$(date '+%F %T') error in export stage1 (see export.log)" > $F/$id.stage1.txt
        log "$id: export stage1 failed, skipped"; continue; fi
    fi
    [ -f $R/3_gvhmr/$id/gvhmr_fast.npz ] && [ -f $R/5_preview/$id.mp4 ] && continue
    ready+=($id)
    [ ${#ready[@]} -ge $CHUNK ] && run_chunk
  done
  run_chunk
  [ -f $W/STOP ] && { log "gvhmr lane stopped"; exit 1; }
  log "gvhmr lane DONE"; touch $W/GVHMR_DONE ;;
watchdog)
  [ -f $W/start_time ] || date +%s > $W/start_time      # wall time counts from the first start, across restarts
  t0=$(cat $W/start_time); last_prog=0
  log "watchdog start (gpu $GPU, max ${MAX_MIB} MiB, ${MAX_H} h)"
  while [ ! -f $W/GVHMR_DONE ]; do
    if [ -f $W/STOP ]; then                            # a lane asked to stop (out of memory / failures): end our processes
      log "watchdog: STOP set by a lane ($(head -1 $W/STOP)), ending our processes"
      pkill -f "$OURS"; sleep 5; pkill -9 -f "$OURS"; break
    fi
    reason=""
    tot=0
    while IFS=, read pid mem; do
      pid=${pid// /}; mem=${mem//[^0-9]/}; [ -z "$pid" ] && continue; mem=${mem:-0}
      args=$(ps -o args= -p $pid 2>/dev/null)
      [ -z "$args" ] && continue                      # process already gone (or not visible): ignore
      if [[ "$args" != *"$W"* && "$args" != *"$R/1_tracking"* ]]; then
        sleep 10; nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | grep -qw "$pid" && \
          foreign="foreign process on card: pid $pid $(ps -o user= -p $pid) ${args:0:80}"
        continue                                       # not ours: not counted in our memory
      fi
      tot=$((tot + mem))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $GPU)
    [ $tot -gt $MAX_MIB ] && reason="our memory ${tot} MiB > ${MAX_MIB}"
    if [ -n "$foreign" ]; then
      if [ "$FOREIGN_STOP" = 1 ]; then reason="$foreign"
      elif [ "$foreign" != "$last_foreign" ]; then log "note (not stopping, FOREIGN_STOP=0): $foreign"; fi
      last_foreign=$foreign
    fi; foreign=""
    free_gb=$(df -BG --output=avail $R | tail -1 | tr -dc 0-9)
    [ "$free_gb" -lt "$MIN_FREE_GB" ] && reason="free disk ${free_gb} GB < ${MIN_FREE_GB} GB"
    [ $(( $(date +%s) - t0 )) -gt $((MAX_H * 3600)) ] && reason="wall time > ${MAX_H} h"
    if [ -n "$reason" ]; then
      log "STOP: $reason"; echo "watchdog $reason" > $W/STOP
      pkill -f "$OURS"; sleep 5; pkill -9 -f "$OURS"
      break
    fi
    echo "$(date '+%T') ${tot}MiB" > $W/watchdog.last
    if [ $(( $(date +%s) - last_prog )) -ge 600 ] && [ -n "$PROGRESS_CMD" ]; then $PROGRESS_CMD >/dev/null 2>&1; last_prog=$(date +%s); fi
    sleep 60
  done
  log "watchdog end" ;;
esac
