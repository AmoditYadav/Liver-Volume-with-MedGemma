"""
Stage A — Vision Specialist: SwinUNETR Segmentation.

Loads the trained SwinUNETR checkpoint, preprocesses a NIfTI CT volume,
runs sliding-window inference, and returns a binary liver mask plus metadata.
"""

import os
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
)


# ─── Config ──────────────────────────────────────────────────────────────────

CHECKPOINT = r"D:\Monai\results\liver_swinunetr\best_metric_model.pth"
SPATIAL_SIZE = (160, 160, 96)          # must match training
ROI_SIZE     = (96, 96, 96)            # sliding-window patch
SW_OVERLAP   = 0.5
A_MIN, A_MAX = -200, 200              # HU clipping range


# ─── Transforms (match preprocess_monai.py) ──────────────────────────────────

def _build_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=A_MIN, a_max=A_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


# ─── Model singleton ─────────────────────────────────────────────────────────

_model = None
_device = None


def _get_model():
    """Lazy-load SwinUNETR onto the best available device (cached)."""
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = SwinUNETR(
        img_size=SPATIAL_SIZE,
        in_channels=1, out_channels=2,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
        feature_size=48,
        norm_name="instance", use_checkpoint=False, spatial_dims=3,
    ).to(_device)

    state = torch.load(CHECKPOINT, map_location=_device, weights_only=True)
    _model.load_state_dict(state)
    _model.eval()
    print(f"[LiverLens] SwinUNETR loaded on {_device}")
    return _model, _device


# ─── Internal: Pre-resize transforms (for effective spacing calc) ────────────

def _build_pre_resize_transforms():
    """Transforms up to Spacingd (before Resized) — used to get resampled shape."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=A_MIN, a_max=A_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=False),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


# ─── Public API ──────────────────────────────────────────────────────────────

def segment(nifti_path: str) -> dict:
    """
    Run liver segmentation on a NIfTI CT volume.

    Returns
    -------
    dict with keys:
        mask_3d          : np.ndarray (H, W, D) — binary liver mask (0/1)
        volume_3d        : np.ndarray (H, W, D) — normalised CT volume [0, 1]
        voxel_spacing    : tuple (sx, sy, sz) in mm — from original NIfTI header
        effective_spacing: tuple (sx, sy, sz) in mm — actual voxel size after
                           Spacingd + Resized (for accurate volume calculation)
        n_slices         : int — number of axial slices
    """
    model, device = _get_model()

    # Read voxel spacing from the original NIfTI header
    nii = nib.load(nifti_path)
    voxel_spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])

    # Step 1: run pre-resize transforms to get the resampled shape
    pre_resize_tfm = _build_pre_resize_transforms()
    pre_data = pre_resize_tfm({"image": nifti_path})
    resampled_shape = pre_data["image"].shape[1:]  # (H', W', D') after 1mm spacing

    # Step 2: compute effective voxel size after Resized
    # After Spacingd(pixdim=1mm) each voxel is 1mm³, but Resized changes dims
    effective_spacing = tuple(
        float(resampled_shape[i]) / float(SPATIAL_SIZE[i])
        for i in range(3)
    )

    # Step 3: full transform pipeline (including Resized)
    transforms = _build_transforms()
    data = transforms({"image": nifti_path})
    tensor = data["image"].unsqueeze(0).to(device)      # [1, 1, H, W, D]

    # Inference
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = sliding_window_inference(
                tensor, roi_size=ROI_SIZE,
                sw_batch_size=2, predictor=model, overlap=SW_OVERLAP,
            )
    mask = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # (H, W, D)
    vol  = tensor[0, 0].cpu().numpy()                     # (H, W, D)

    print(f"[LiverLens] Resampled shape: {resampled_shape} → "
          f"Resized to: {SPATIAL_SIZE} → "
          f"Effective spacing: {tuple(f'{s:.2f}' for s in effective_spacing)} mm")

    return {
        "mask_3d":           mask.astype(np.uint8),
        "volume_3d":         vol,
        "voxel_spacing":     voxel_spacing,
        "effective_spacing":  effective_spacing,
        "n_slices":          mask.shape[2],
    }
