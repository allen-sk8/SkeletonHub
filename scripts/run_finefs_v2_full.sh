#!/bin/bash
# FineFS v2 over the whole dataset (1,167 videos), one GPU, resumable. Run in tmux:
#   GPU=5 bash scripts/run_finefs_v2_full.sh
# Supervisor: works out which videos still need doing, runs the three lanes of run_finefs_v2_batch.sh (track, gvhmr,
# watchdog) on one card, waits, and decides from $W/STOP what to do:
#   "oom ..."  out of memory. If another user's process is on the card, the video is retried on another empty card
#              (picked with ~/hq/bin/gpufree; if none is free, wait 30 min and ask again). If no other process is there,
#              the video really needs too much memory: it stays recorded as failed and the batch goes on on the same card;
#              3 such cases in a row stop the batch.
#   anything else (our memory too high, disk nearly full, repeated failures, wall time): the batch stops, reason logged.
# Settings (Allen 2026-11-24): BoT-SORT with Re-ID + offline lock-on, half-resolution crops, previews for every video,
# FOREIGN_STOP=0 (do not stop when a colleague's process appears on the card).
cd /home/allen/SkeletonHub
export W=${W:-outputs/finefs_v2_full} R=${R:-/home/allen/datasets/FineFS}
export FOREIGN_STOP=${FOREIGN_STOP:-0} MAX_H=${MAX_H:-384} MAX_MIB=${MAX_MIB:-14000} MIN_FREE_GB=${MIN_FREE_GB:-300}
GPU=${GPU:?set GPU=<nvidia-smi index>}
PY=/home/allen/miniconda3/envs/reidtrack/bin/python
export PROGRESS_CMD="$PY scripts/finefs_v2_progress.py --work $W --root $R"
mkdir -p $W
log() { echo "[$(date '+%F %T')] [supervisor] $*" | tee -a $W/batch.log; }
ls /home/allen/datasets/FineFS/video | sed -n 's/\.mp4$//p' | sort -n > $W/all_ids.txt
oom_here=0; switches=0
while true; do
  # still to do: not finished in R, not failed, not "no person"
  todo=()
  while read id; do
    [ -f $R/3_gvhmr/$id/gvhmr_fast.npz ] && [ -f $R/5_preview/$id.mp4 ] && [ -f $R/4_flags/$id/flags.npz ] && continue
    ls $W/_failed/$id.*.txt >/dev/null 2>&1 && continue
    [ -f $W/_track/$id/track_stats.json ] && grep -q '"no_person": true' $W/_track/$id/track_stats.json && continue
    todo+=($id)
  done < $W/all_ids.txt
  if [ ${#todo[@]} -eq 0 ]; then
    log "nothing left to do; writing the summary checks"
    $PY scripts/finefs_v2_export.py stage4 --root $R --summary-dir $W --ids $(paste -sd, $W/all_ids.txt) --fail-dir $W/_failed 2>&1 \
      | grep -v '^\[check\]' | tee -a $W/export.log
    touch $W/ALL_DONE; echo "finished $(date '+%F %T')" > $W/supervisor.last; $PROGRESS_CMD >/dev/null; log "ALL DONE"; break
  fi
  export GPU IDS=$(IFS=,; echo "${todo[*]}")
  rm -f $W/STOP $W/GVHMR_DONE $W/_track/ALL_DONE
  touch $W/run_start
  echo "card $GPU, ${#todo[@]} videos to do, started $(date '+%F %T')" > $W/supervisor.last
  log "start on GPU $GPU: ${#todo[@]} videos to do"
  bash scripts/run_finefs_v2_batch.sh watchdog & wd=$!
  bash scripts/run_finefs_v2_batch.sh track & tr=$!
  bash scripts/run_finefs_v2_batch.sh gvhmr & gv=$!
  wait $tr $gv
  [ -f $W/GVHMR_DONE ] || [ -f $W/STOP ] || echo "lanes ended without a reason" > $W/STOP
  wait $wd
  $PROGRESS_CMD >/dev/null 2>&1
  reason=$(head -1 $W/STOP 2>/dev/null)
  if [ -z "$reason" ]; then                                      # gvhmr lane done: loop once more to check what is left
    if [ "$IDS" = "$prev_ids" ]; then
      echo "stopped: a full pass changed nothing for ${#todo[@]} videos (see batch.log WARNING lines)" > $W/supervisor.last
      log "STOP: a full pass changed nothing for: $IDS"; break
    fi
    prev_ids=$IDS; oom_here=0; continue
  fi
  case "$reason" in
  oom*)
    sleep 10
    others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU | tr -d ' ')
    marker=$(grep -l '^.* oom$' $(find $W/_failed -name '*.txt' -newer $W/run_start 2>/dev/null) /dev/null 2>/dev/null | head -1)
    if [ -n "$others" ]; then
      who=$(for p in $others; do echo -n "$p:$(ps -o user= -p $p) "; done)
      log "out of memory with another process on GPU $GPU ($who); retry ${marker:+$(basename $marker) }on another card"
      [ -n "$marker" ] && rm -f "$marker"
      oom_here=0; switches=$((switches + 1))
      while true; do
        new=$(~/hq/bin/gpufree | sed -n 's/.*CUDA_VISIBLE_DEVICES=\([0-9]*\).*/\1/p')
        [ -n "$new" ] && [ "$new" != "$GPU" ] && break
        echo "waiting for an empty card since $(date '+%F %T')" > $W/supervisor.last
        log "no empty card that may be used; waiting 30 min"; sleep 1800
      done
      log "switching GPU $GPU -> $new (switch $switches)"; GPU=$new
    else
      oom_here=$((oom_here + 1))
      log "out of memory with no other process on the card: $(basename "$marker") stays failed ($oom_here in a row)"
      [ $oom_here -ge 3 ] && { echo "stopped: 3 out-of-memory failures in a row on an empty card" > $W/supervisor.last; log "STOP"; break; }
    fi ;;
  *)
    echo "stopped: $reason ($(date '+%F %T'))" > $W/supervisor.last
    log "STOP for good: $reason"; break ;;
  esac
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i $GPU | tee -a $W/batch.log
