# HybrIK Model Checkpoint & Parameter Setup Instructions

To run the native HybrIK 3D Pose Estimator in SkeletonHub, you need to download the pre-trained weights and the auxiliary SMPL/SMPLX body model parameters. Follow the instructions below to download and place them in the correct directories.

---

## 1. HybrIK HRNet Pretrained Checkpoint

Download the main model checkpoint `hybrik_hrnet.pth`:
- **Source Link**: [HybrIK Model Weights (Google Drive / GitHub Releases)](https://github.com/jeffas6/HybrIK#model-zoo)
- **Target Path**: `common_models/checkpoints/hybrik/hybrik_hrnet.pth`

```bash
# Example directory structure:
# common_models/checkpoints/hybrik/hybrik_hrnet.pth
```

---

## 2. SMPL / SMPLX Body Model Parameters

HybrIK requires standard SMPL and SMPLX model template files to construct and regress body joint coordinates (e.g., SMPL24, OpenPose BODY25).

Place these files in `common_models/body_models/smpl/`:

### A. Core SMPL Models
- `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
- `h36m_mean_beta.npy`
- `J_regressor_h36m.npy`
- `smplx_kid_template.npy`

### B. Flat SMPLX Folder
Place the following files in `common_models/body_models/smpl/smplx/`:
- `SMPLX_FEMALE.npz`
- `SMPLX_FEMALE.pkl`
- `SMPLX_MALE.npz`
- `SMPLX_MALE.pkl`
- `SMPLX_NEUTRAL.npz`
- `SMPLX_NEUTRAL.pkl`

---

## 3. How to Run local Inference
Once the checkpoint and parameters are in place, the detector can be called directly from Python:

```python
from detectors.hybrik_detector import HybrikDetector

detector = HybrikDetector(device='cuda:0')
results = detector.detect_video('path/to/video.mp4')
# results['smpl24'] contains shape (T, 24, 3)
# results['body25'] contains shape (T, 25, 3)
```
