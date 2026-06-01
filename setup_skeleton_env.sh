#!/bin/bash
# Exit on error
set -e

echo "========================================="
echo "  Starting SkeletonHub Env Installation  "
echo "========================================="

# Clean any existing environment
echo "--> 1. Removing old skeleton_env..."
conda env remove -n skeleton_env -y || true

# Create brand new environment
echo "--> 2. Creating clean skeleton_env (Python 3.10)..."
conda create -n skeleton_env python=3.10 pip -y

# Assign binary paths
PIP="/home/allen/miniconda3/envs/skeleton_env/bin/pip"
PYTHON="/home/allen/miniconda3/envs/skeleton_env/bin/python"

# Verify pip is native to the environment
echo "--> Active pip version and path:"
$PIP -V

echo "--> 3. Installing PyTorch & CUDA 11.7..."
$PIP install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo "--> 4. Installing PyTorch3D..."
$PIP install fvcore iopath
$PIP install pytorch3d==0.7.8 --find-links https://dl.fbaipublicfiles.com/pytorch3d/prefixs/wheels/py310_cu117_pyt201/packages.html

echo "--> 5. Installing MMCV (using OpenMMLab 2.0.1 prebuilts)..."
$PIP install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html

echo "--> 6. Installing chumpy (without build isolation)..."
$PIP install --no-build-isolation git+https://github.com/mattloper/chumpy

echo "--> 7. Installing SkeletonHub Main requirements..."
$PIP install -r /home/allen/SkeletonHub/requirements.txt

echo "--> 8. Installing WHAM requirements..."
$PIP install -r /home/allen/SkeletonHub/external/WHAM/requirements.txt

echo "--> 9. Installing ViTPose package..."
cd /home/allen/SkeletonHub/external/WHAM/third-party/ViTPose
$PIP install -e .

echo "========================================="
echo "  Environment Rebuilt Successfully!     "
echo "========================================="
