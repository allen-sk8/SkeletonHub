#!/usr/bin/env bash
# Build the clip directories used for the HMR zoo comparison (see docs/HMR_ZOO.md).
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free gpu> bash scripts/hmr_zoo_prepare_clips.sh
# Tracking needs the `reidtrack` env (uses <=4 GB GPU memory).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/allen/miniconda3/envs/reidtrack/bin/python
PREP="nice -n 10 $PY -m hmr_zoo.prep_clip"
RAW=/home/allen/skating_jump_clips/data/raw_videos
HAA=/home/allen/datasets/MoVid/videos/haa500
FD=/home/allen/datasets/FineDiving/0_clips

# figure skating (FineFS Stage 2 clips; boxes copied from finefs_preproc)
$PREP --finefs 4Lz_4Lz_0018 --sport figure_skating
# roller skating (Allen's own practice videos; v00 = labelled 3F, v25 = labelled Salchow at 14.9-19.2 s)
$PREP --video $RAW/v00.mp4 --start 0 --dur 4.6 --sport roller_skating --name v00_3F_0s --note "Allen label: 3F"
$PREP --video $RAW/v25.mp4 --start 14.2 --dur 7 --sport roller_skating --name v25_S_14s --note "Allen label: Salchow at 14.9-19.2 s"
# diving (FineDiving, 456x256, 3-5 s clips)
$PREP --video $FD/101b/101b_0001/101b_0001.mp4 --start 0 --dur 10 --sport diving --name 101b_0001
$PREP --video $FD/107b/107b_0137/107b_0137.mp4 --start 0 --dur 10 --sport diving --name 107b_0137
# athletics (HAA500 via MoVid; clips are only 1.5-3 s)
$PREP --video $HAA/video_splits3/triple_jump_jump_12_clip1.mp4 --start 0 --dur 10 --sport athletics --name triple_jump_12
$PREP --video $HAA/video_splits1/javelin_throw_9_clip1.mp4 --start 0 --dur 10 --sport athletics --name javelin_throw_9
# baseball pitching (no video in openbiomechanics; HAA500 pitch clips instead)
$PREP --video $HAA/video_splits0/baseball_pitch_21_clip1.mp4 --start 0 --dur 10 --sport baseball --name pitch_21
$PREP --video $HAA/video_splits0/baseball_pitch_11_clip1.mp4 --start 0 --dur 10 --sport baseball --name pitch_11
# boxing proxy (BX_2026 raw video lives in a colleague's folder; HAA500 sandbag punching used instead, NOT real boxing)
$PREP --video $HAA/video_splits2/punching_sandbag_8_clip1.mp4 --start 0 --dur 10 --sport boxing_proxy --name sandbag_8
$PREP --video $HAA/video_splits2/punching_sandbag_clip1.mp4 --start 0 --dur 10 --sport boxing_proxy --name sandbag_1
