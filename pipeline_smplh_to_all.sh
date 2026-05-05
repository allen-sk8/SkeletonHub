#!/bin/bash
# Pipeline: SMPL-H (.pkl) -> 52j, 24j, 22j, 263d
set -e

if [ -z "$1" ]; then
    echo "Usage: ./pipeline_smplh_to_all.sh <path_to_smplh_pkl>"
    exit 1
fi

INPUT_FILE=$1
# Ensure basename is stripped of any possible prefix if user passed a generated file, or just use the whole basename.
BASENAME=$(basename "$INPUT_FILE" .pkl)
# Strip smplh_ prefix if it exists so we don't end up with 52j_smplh_xxx
BASENAME=${BASENAME#smplh_}

J52_FILE="data/smpl_joints/samples_52j/52j_${BASENAME}.npy"
J24_FILE="data/smpl_joints/samples_24j/24j_${BASENAME}.npy"
J22_FILE="data/smpl_joints/samples_22j/22j_${BASENAME}.npy"
FEAT263_FILE="data/smpl_joints/samples_263d/263d_${BASENAME}.npy"

echo "========================================================="
echo "🚀 [Step 1] SMPL-H (.pkl) -> 52j (.npy)"
echo "========================================================="
python converters/smplh_to_smplh_52j.py "$INPUT_FILE" --output "$J52_FILE"
python visualizers/vis_smpl_joints.py "$J52_FILE"

echo "========================================================="
echo "🚀 [Step 2] SMPL-H (.pkl) -> 24j (.npy)"
echo "========================================================="
python converters/smplh_to_smpl_24j.py "$INPUT_FILE" --output "$J24_FILE"
python visualizers/vis_smpl_joints.py "$J24_FILE"

echo "========================================================="
echo "🚀 [Step 3] SMPL-H (.pkl) -> 22j (.npy)"
echo "========================================================="
python converters/smplh_to_humanml3d_22j.py "$INPUT_FILE" --output "$J22_FILE"
python visualizers/vis_smpl_joints.py "$J22_FILE"

echo "========================================================="
echo "🚀 [Step 4] 22j (.npy) -> 263D features (.npy)"
echo "========================================================="
python converters/humanml3d_22j_to_humanml3d_263d.py "$J22_FILE" --output "$FEAT263_FILE"

echo "========================================================="
echo "🚀 [Step 5] Visualizing Original SMPL-H Mesh"
echo "========================================================="
python visualizers/vis_smplh_mesh.py "$INPUT_FILE"

echo "========================================================="
echo "✅ Pipeline Complete!"
echo "Outputs & Visualizations:"
echo " - [52j] $J52_FILE ➔ data/smpl_joints/samples_52j/visualizations/vis_52j_52j_${BASENAME}.mp4"
echo " - [24j] $J24_FILE ➔ data/smpl_joints/samples_24j/visualizations/vis_24j_24j_${BASENAME}.mp4"
echo " - [22j] $J22_FILE ➔ data/smpl_joints/samples_22j/visualizations/vis_22j_22j_${BASENAME}.mp4"
echo " - [Feat] $FEAT263_FILE"
echo " - [Mesh] data/smpl/smplh/visualizations/vis_smplh_${BASENAME}.mp4"
echo "========================================================="
