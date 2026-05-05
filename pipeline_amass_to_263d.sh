#!/bin/bash
# Pipeline: AMASS (.npz) -> SMPL-H (.pkl) -> HumanML3D 22j (.npy) -> HumanML3D 263d (.npy)
set -e

if [ -z "$1" ]; then
    echo "Usage: ./pipeline_amass_to_263d.sh <path_to_amass_npz>"
    exit 1
fi

INPUT_FILE=$1
BASENAME=$(basename "$INPUT_FILE" .npz)
SMPLH_FILE="data/smpl/smplh/smplh_${BASENAME}.pkl"
J22_FILE="data/smpl_joints/samples_22j/22j_${BASENAME}.npy"
FEAT263_FILE="data/smpl_joints/samples_263d/263d_${BASENAME}.npy"

echo "========================================================="
echo "🚀 [Step 1] AMASS (.npz) -> SMPL-H (.pkl)"
echo "========================================================="
python converters/amass_to_smplh.py "$INPUT_FILE" --output "$SMPLH_FILE"
python visualizers/vis_smplh_mesh.py "$SMPLH_FILE"

echo "========================================================="
echo "🚀 [Step 2] SMPL-H (.pkl) -> HumanML3D 22j (.npy)"
echo "========================================================="
python converters/smplh_to_humanml3d_22j.py "$SMPLH_FILE" --output "$J22_FILE"
python visualizers/vis_smpl_joints.py "$J22_FILE"

echo "========================================================="
echo "🚀 [Step 3] HumanML3D 22j (.npy) -> HumanML3D 263d features (.npy)"
echo "========================================================="
python converters/humanml3d_22j_to_humanml3d_263d.py "$J22_FILE" --output "$FEAT263_FILE"
# Note: 263D is a feature vector, usually we visualize the source 22j joints instead.

echo "========================================================="
echo "✅ Pipeline Complete!"
echo "Outputs & Visualizations:"
echo " - [Params]  $SMPLH_FILE ➔ data/smpl/smplh/visualizations/vis_smplh_smplh_${BASENAME}.mp4"
echo " - [Joints]  $J22_FILE ➔ data/smpl_joints/samples_22j/visualizations/vis_22j_22j_${BASENAME}.mp4"
echo " - [Feature] $FEAT263_FILE"
echo "========================================================="
