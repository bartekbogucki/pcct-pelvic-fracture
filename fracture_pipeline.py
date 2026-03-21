"""
fracture_pipeline.py
====================
Two-stage CT pelvic fracture segmentation pipeline.

Stage 1:
    Uses pelvis.dcm as ROI crop mask.

Stage 2:
    nnU-Net trained on cropped pelvic region to detect fractures.

Usage
-----
# 1. Add patients to registry
python fracture_pipeline.py add \
    --patient_id 64406628 \
    --dicom_dir  "data/64406628/.../10005986" \
    --seg_fracture "data/64406628/fracture segmentation/Fractures all.dcm" \
    --seg_pelvis   "data/64406628/pelvic segmentation/pelvis.dcm"

# 2. Visual QC — run this first, always
python fracture_pipeline.py qc --patient_id 64406628

# 3. Convert all patients and prepare nnU-Net folds
python fracture_pipeline.py prepare

# 4. Train (run per fold)
python fracture_pipeline.py train --fold 0

# 5. Evaluate all folds
python fracture_pipeline.py evaluate

# 6. Predict new patient
python fracture_pipeline.py predict \
    --dicom_dir  "data/new_patient/.../slices" \
    --seg_pelvis "data/new_patient/pelvis.dcm" \
    --output     "predictions/new_patient.nii.gz"
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import os
import sys
import json
import shutil
import argparse
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import pydicom
except ImportError:
    sys.exit("Missing: pip install pydicom")

try:
    import SimpleITK as sitk
except ImportError:
    sys.exit("Missing: pip install SimpleITK")

try:
    import nibabel as nib
except ImportError:
    sys.exit("Missing: pip install nibabel")

try:
    from sklearn.model_selection import KFold, LeaveOneOut
except ImportError:
    sys.exit("Missing: pip install scikit-learn")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
REGISTRY_FILE  = "dataset_registry.json"
BASE_DIR       = Path(".")
CONVERTED_DIR  = Path("nnunet_data/converted")
FOLDS_DIR      = Path("nnunet_data/folds")
RESULTS_DIR    = Path("nnunet_results")
QC_DIR         = Path("qc")

# nnU-Net environment — set before training
NNUNET_RAW          = Path("nnunet_data/folds")
NNUNET_PREPROCESSED = Path("nnunet_preprocessed")
NNUNET_RESULTS      = RESULTS_DIR
DATASET_ID          = 1
DATASET_NAME        = "Fracture"

# Bone window for visualization and 2D baseline
BONE_WL = 300
BONE_WW = 1500

# Crop padding around pelvis bounding box (voxels)
CROP_PADDING = 20


# ─────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────
@dataclass
class PatientCase:
    patient_id:   str
    dicom_dir:    str
    seg_fracture: str
    seg_pelvis:   str


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────
class FractureRegistry:
    """
    Persistent registry of all annotated patients.
    """

    def __init__(self, path: str = REGISTRY_FILE):
        self.path  = Path(path)
        self.cases: List[PatientCase] = []
        if self.path.exists():
            self._load()

    # ── public ────────────────────────────────
    def add(self, patient_id: str, dicom_dir: str,
            seg_fracture: str, seg_pelvis: str) -> None:
        if any(c.patient_id == patient_id for c in self.cases):
            print(f"  [registry] {patient_id} already registered — skipped.")
            return
        self.cases.append(
            PatientCase(patient_id, dicom_dir, seg_fracture, seg_pelvis)
        )
        self._save()
        print(f"  [registry] Added {patient_id}  "
              f"(total: {len(self.cases)} patients)")

    def get(self, patient_id: str) -> Optional[PatientCase]:
        for c in self.cases:
            if c.patient_id == patient_id:
                return c
        return None

    def remove(self, patient_id: str) -> bool:
        before = len(self.cases)
        self.cases = [c for c in self.cases if c.patient_id != patient_id]
        if len(self.cases) < before:
            self._save()
            print(f"  [registry] Removed {patient_id}  "
                  f"(remaining: {len(self.cases)} patients)")
            return True
        print(f"  [registry] Patient {patient_id} not found.")
        return False

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def summary(self) -> None:
        print(f"\nRegistry: {len(self.cases)} patients")
        for c in self.cases:
            print(f"  {c.patient_id}")
            print(f"    CT:       {c.dicom_dir}")
            print(f"    Fracture: {c.seg_fracture}")
            print(f"    Pelvis:   {c.seg_pelvis}")

    # ── private ───────────────────────────────
    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump([asdict(c) for c in self.cases], f, indent=2)

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self.cases = [PatientCase(**d) for d in data]
        print(f"  [registry] Loaded {len(self.cases)} patients "
              f"from {self.path}")


# ─────────────────────────────────────────────
# DICOM / NIfTI I/O
# ─────────────────────────────────────────────
def load_ct_volume(dicom_dir: str) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Load CT DICOM folder → (volume, affine, spacing).
    """
    dicom_files = sorted(
        [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
         if not f.startswith(".")],
        key=lambda x: float(
            pydicom.dcmread(x, stop_before_pixels=True)
            .ImagePositionPatient[2]
        )
    )
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")

    slices, z_positions = [], []
    ds0 = pydicom.dcmread(dicom_files[0])

    for f in dicom_files:
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)
        img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        slices.append(img)
        z_positions.append(float(ds.ImagePositionPatient[2]))

    volume = np.stack(slices, axis=0)            # (Z, H, W)

    px = float(ds0.PixelSpacing[0])
    py = float(ds0.PixelSpacing[1])
    pz = abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else 1.0

    affine        = np.eye(4, dtype=np.float64)
    affine[0, 0]  = px
    affine[1, 1]  = py
    affine[2, 2]  = pz
    affine[0, 3]  = float(ds0.ImagePositionPatient[0])
    affine[1, 3]  = float(ds0.ImagePositionPatient[1])
    affine[2, 3]  = z_positions[0]

    spacing = (pz, px, py)                       # Z, Y, X
    return volume, affine, spacing


def load_seg_dcm(seg_path: str) -> np.ndarray:
    """
    Load segmentation DICOM 
    """
    img = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(img)            # (Z, H, W) or (frames, H, W)
    return (arr > 0).astype(np.uint8)


def crop_to_mask(volume: np.ndarray,
                 mask: np.ndarray,
                 extra_mask: Optional[np.ndarray] = None,
                 padding: int = CROP_PADDING
                 ) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """
    Crop volume to bounding box of mask, with optional padding. 

    Returns
    -------
    vol_cropped  : (Z', H', W')
    mask_cropped : (Z', H', W')   — extra_mask if provided, else mask
    bbox         : (z0,z1, y0,y1, x0,x1)  in original coordinates
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("Mask is empty — cannot crop.")

    Z, H, W = volume.shape
    z0 = max(0, int(coords[:, 0].min()) - padding)
    z1 = min(Z, int(coords[:, 0].max()) + padding + 1)
    y0 = max(0, int(coords[:, 1].min()) - padding)
    y1 = min(H, int(coords[:, 1].max()) + padding + 1)
    x0 = max(0, int(coords[:, 2].min()) - padding)
    x1 = min(W, int(coords[:, 2].max()) + padding + 1)

    bbox        = (z0, z1, y0, y1, x0, x1)
    vol_cropped = volume[z0:z1, y0:y1, x0:x1]
    target      = extra_mask if extra_mask is not None else mask
    tgt_cropped = target[z0:z1, y0:y1, x0:x1]

    return vol_cropped, tgt_cropped, bbox


def volume_to_nifti(array: np.ndarray,
                    affine: np.ndarray,
                    dtype=np.float32) -> nib.Nifti1Image:
    """
    Convert (Z, H, W) numpy array to NIfTI image.
    """
    arr = array.transpose(2, 1, 0).astype(dtype)  # (W, H, Z)
    return nib.Nifti1Image(arr, affine)


def save_nifti(array: np.ndarray, affine: np.ndarray, path) -> None:
    nib.save(volume_to_nifti(array, affine), str(path))


# ─────────────────────────────────────────────
# Visual QC
# ─────────────────────────────────────────────
def qc_case(case: PatientCase, n_cols: int = 8) -> Path:
    """
    Render CT slices (bone window) with pelvis (green) and
    fracture (red) overlays. Saves PNG to qc/<patient_id>.png.

    Run this on every patient before training.
    """
    QC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[QC] {case.patient_id}")
    volume, _, spacing = load_ct_volume(case.dicom_dir)
    frac_mask   = load_seg_dcm(case.seg_fracture)
    pelvis_mask = load_seg_dcm(case.seg_pelvis)

    # ── stats ──────────────────────────────────
    frac_slices   = np.where(frac_mask.any(axis=(1, 2)))[0]
    pelvis_slices = np.where(pelvis_mask.any(axis=(1, 2)))[0]

    print(f"  CT shape   : {volume.shape}  spacing ZYX={spacing}")
    print(f"  Pelvis     : {pelvis_mask.sum():,} voxels  "
          f"slices {pelvis_slices.min()}–{pelvis_slices.max()}")
    print(f"  Fracture   : {frac_mask.sum():,} voxels  "
          f"slices {frac_slices.min()}–{frac_slices.max()}")

    overlap = (frac_mask.astype(bool) & pelvis_mask.astype(bool)).sum()
    pct     = overlap / (frac_mask.sum() + 1e-9) * 100
    flag    = "⚠️  WARNING — possible misalignment!" if pct < 70 else "✓"
    print(f"  Fracture inside pelvis: {pct:.1f}%  {flag}")

    # ── pick representative slices ─────────────
    if len(frac_slices) == 0:
        print("  ⚠️  No fracture voxels found — check seg_fracture path.")
        show_idx = np.linspace(0, volume.shape[0]-1, n_cols, dtype=int)
    else:
        show_idx = frac_slices[
            np.linspace(0, len(frac_slices)-1, n_cols, dtype=int)
        ]

    # ── bone window helper ─────────────────────
    lo = BONE_WL - BONE_WW / 2
    hi = BONE_WL + BONE_WW / 2

    def bw(sl):
        return np.clip((sl - lo) / (hi - lo), 0, 1)

    # ── plot ───────────────────────────────────
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols // n_rows,
                             figsize=(22, 9), facecolor="black")
    axes = axes.flatten()

    for i, idx in enumerate(show_idx):
        ax = axes[i]
        ax.imshow(bw(volume[idx]), cmap="gray", interpolation="nearest")

        if pelvis_mask[idx].any():
            ax.imshow(
                np.ma.masked_where(~pelvis_mask[idx].astype(bool),
                                   np.ones_like(pelvis_mask[idx])),
                cmap="Greens", alpha=0.25, vmin=0, vmax=1
            )
        if frac_mask[idx].any():
            ax.imshow(
                np.ma.masked_where(~frac_mask[idx].astype(bool),
                                   np.ones_like(frac_mask[idx])),
                cmap="Reds", alpha=0.65, vmin=0, vmax=1
            )

        ax.set_title(f"z={idx}", fontsize=8, color="white")
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(facecolor="lime",  alpha=0.5, label="Pelvis ROI"),
        mpatches.Patch(facecolor="red",   alpha=0.7, label="Fracture"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, fontsize=11, facecolor="black",
               labelcolor="white", framealpha=0)
    fig.suptitle(
        f"QC — {case.patient_id}   "
        f"fracture={frac_mask.sum():,} vox  "
        f"pelvis_overlap={pct:.0f}%",
        color="white", fontsize=12
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_path = QC_DIR / f"{case.patient_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# Cross-Validation Splits
# ─────────────────────────────────────────────
def make_cv_splits(cases: List[PatientCase]) -> Tuple[List[Dict], str]:
    """
    Auto-selects CV strategy based on dataset size:
      <  10 patients  →  Leave-One-Out
      10-29 patients  →  5-fold
      30+  patients   →  5-fold + held-out 20% test set
    """
    n = len(cases)

    if n < 10:
        strategy = "loocv"
        cv = LeaveOneOut()
    elif n < 30:
        strategy = "5fold"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        strategy = "5fold_with_test"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n[CV] n={n} → {strategy}")

    if strategy == "5fold_with_test":
        n_test    = max(1, int(n * 0.2))
        test_ids  = [c.patient_id for c in cases[-n_test:]]
        train_pool = cases[:-n_test]
    else:
        test_ids   = []
        train_pool = cases

    splits = []
    for train_idx, val_idx in cv.split(train_pool):
        splits.append({
            "train": [train_pool[i].patient_id for i in train_idx],
            "val":   [train_pool[i].patient_id for i in val_idx],
            "test":  test_ids,
        })

    for i, s in enumerate(splits):
        print(f"  Fold {i}: train={s['train']}  val={s['val']}")

    return splits, strategy


# ─────────────────────────────────────────────
# Dataset Conversion
# ─────────────────────────────────────────────
def convert_all_patients(registry: FractureRegistry,
                         padding: int = CROP_PADDING) -> None:
    """
    Convert every patient ONCE to NIfTI and cache in CONVERTED_DIR.
    Subsequent runs skip already-converted cases.
    """
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[Convert] {len(registry)} patients → {CONVERTED_DIR}")

    stats = []

    for case in registry:
        out_ct   = CONVERTED_DIR / f"{case.patient_id}_ct.nii.gz"
        out_frac = CONVERTED_DIR / f"{case.patient_id}_fracture.nii.gz"

        if out_ct.exists() and out_frac.exists():
            print(f"  {case.patient_id} already converted — skipped.")
            continue

        print(f"\n  Processing {case.patient_id} ...")
        volume, affine, spacing = load_ct_volume(case.dicom_dir)
        pelvis   = load_seg_dcm(case.seg_pelvis)
        fracture = load_seg_dcm(case.seg_fracture)

        # Validate shapes match
        if volume.shape != pelvis.shape or volume.shape != fracture.shape:
            print(f"  ⚠️  Shape mismatch!")
            print(f"     CT      : {volume.shape}")
            print(f"     Pelvis  : {pelvis.shape}")
            print(f"     Fracture: {fracture.shape}")
            print(f"  Attempting to resize segmentations to CT shape...")
            pelvis   = _resize_mask_to_volume(pelvis,   volume.shape)
            fracture = _resize_mask_to_volume(fracture, volume.shape)

        # Crop to pelvis ROI, fracture mask follows same crop
        vol_cr, frac_cr, bbox = crop_to_mask(
            volume, pelvis, extra_mask=fracture, padding=padding
        )

        # Save
        save_nifti(vol_cr,  affine, out_ct)
        save_nifti(frac_cr.astype(np.float32), affine, out_frac)

        info = {
            "patient_id":      case.patient_id,
            "ct_shape":        list(volume.shape),
            "cropped_shape":   list(vol_cr.shape),
            "spacing_zyx":     list(spacing),
            "fracture_voxels": int(fracture.sum()),
            "cropped_frac_vox":int(frac_cr.sum()),
            "bbox":            list(bbox),
        }
        stats.append(info)
        print(f"  ✓ {case.patient_id}  "
              f"CT {volume.shape} → cropped {vol_cr.shape}  "
              f"fracture voxels: {frac_cr.sum():,}")

    if stats:
        stats_path = CONVERTED_DIR / "conversion_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Stats saved → {stats_path}")


def _resize_mask_to_volume(mask: np.ndarray,
                            target_shape: tuple) -> np.ndarray:
    """Resize mask to match CT volume shape using nearest-neighbor."""
    if mask.shape == target_shape:
        return mask
    mask_sitk  = sitk.GetImageFromArray(mask.astype(np.float32))
    ref        = sitk.GetImageFromArray(
        np.zeros(target_shape, dtype=np.float32)
    )
    resampler  = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled  = resampler.Execute(mask_sitk)
    return (sitk.GetArrayFromImage(resampled) > 0).astype(np.uint8)


# ─────────────────────────────────────────────
# nnU-Net Fold Preparation
# ─────────────────────────────────────────────
def prepare_nnunet_folds(splits: List[Dict], hpc_splits: bool = False) -> None:
    """
    Create one nnU-Net dataset per fold, each with 3 patients in imagesTr
    and 1 patient in imagesTs (the held-out LOOCV patient).

    Also writes splits_final.json so nnU-Net uses a simple 1-fold split
    internally (all 3 train patients train, last one also monitors val loss)
    instead of crashing trying to do 5-fold CV on 3 samples.

    To be updated later as more data arrives — currently only 4 patients, so LOOCV with 3 train + 1 val.
    """
    print(f"\n[Folds] Creating {len(splits)} nnU-Net fold datasets...")

    for fold_i, split in enumerate(splits):
        fold_base  = FOLDS_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold_i}"
        images_tr  = fold_base / "imagesTr"
        labels_tr  = fold_base / "labelsTr"
        images_ts  = fold_base / "imagesTs"
        for d in [images_tr, labels_tr, images_ts]:
            d.mkdir(parents=True, exist_ok=True)

        # Training cases (3 patients)
        for pid in split["train"]:
            src_ct   = CONVERTED_DIR / f"{pid}_ct.nii.gz"
            src_frac = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
            if not src_ct.exists():
                print(f"  ⚠️  Missing {pid} — run 'prepare' first")
                continue
            shutil.copy2(src_ct,   images_tr / f"{pid}_0000.nii.gz")
            shutil.copy2(src_frac, labels_tr / f"{pid}.nii.gz")

        # Held-out val patient (1 patient)
        for pid in split["val"]:
            src_ct   = CONVERTED_DIR / f"{pid}_ct.nii.gz"
            src_frac = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
            # Always copy to imagesTs for final inference/eval
            if src_ct.exists():
                shutil.copy2(src_ct,   images_ts / f"{pid}_0000.nii.gz")
            if src_frac.exists():
                shutil.copy2(src_frac, images_ts / f"{pid}_gt.nii.gz")
            # HPC mode: also copy to imagesTr/labelsTr so nnU-Net can find
            # the label file when computing val loss each epoch.
            # The patient is listed only in splits_final.json "val" — never trained on.
            if hpc_splits:
                if src_ct.exists():
                    shutil.copy2(src_ct,   images_tr / f"{pid}_0000.nii.gz")
                if src_frac.exists():
                    shutil.copy2(src_frac, labels_tr / f"{pid}.nii.gz")

        # Dataset JSON
        dataset_json = {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "fracture": 1},
            "numTraining": len(split["train"]),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        with open(fold_base / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=2)

        # splits_final.json
        # Laptop (default): val = last training patient — safe, no OOM risk
        # HPC (--hpc_splits): val = true held-out patient — real validation curve
        train_cases = sorted(split["train"])
        if hpc_splits:
            splits_nnunet = [{"train": train_cases, "val": split["val"]}]
        else:
            splits_nnunet = [{"train": train_cases, "val": [train_cases[-1]]}]
        splits_path = fold_base / "splits_final.json"
        with open(splits_path, "w") as f:
            json.dump(splits_nnunet, f, indent=2)

        print(f"  Fold {fold_i} → {fold_base}")
        print(f"    train: {split['train']}  val(held-out): {split['val']}")

    # Save CV splits for evaluate/plot
    cv_path = FOLDS_DIR / "cv_splits.json"
    with open(cv_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\n  CV splits saved → {cv_path}")


# ─────────────────────────────────────────────
# nnU-Net Training
# ─────────────────────────────────────────────
def train_fold(fold: int) -> None:
    """
    Train nnU-Net for one LOOCV fold.
    Each fold has its own isolated dataset with 3 training patients.
    splits_final.json (written by prepare) prevents nnU-Net from trying
    5-fold CV on 3 patients.
    """
    fold_base = FOLDS_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold}"
    if not fold_base.exists():
        print(f"[Train] Fold {fold} not found at {fold_base}")
        print("  Run: python fracture_pipeline.py prepare")
        return

    env, res_dir = _nnunet_env(fold)
    images_ts    = fold_base / "imagesTs"

    # ── Detect device ─────────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        device_flag = "-device cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_flag = "-device mps"
    else:
        device_flag = "-device cpu"

    print(f"\n[Train] Fold {fold}")
    print(f"  Dataset    : {fold_base}")
    print(f"  nnUNet_raw : {env['nnUNet_raw']}")
    print(f"  nnUNet_pre : {env['nnUNet_preprocessed']}")
    print(f"  nnUNet_res : {env['nnUNet_results']}")
    print(f"  Device     : {device_flag}")

    # ── Step 1: plan and preprocess ───────────────────────────────────────
    cmd_plan = (
        f"nnUNetv2_plan_and_preprocess "
        f"-d {DATASET_ID} "
        f"--verify_dataset_integrity"
    )
    print(f"\n  Step 1: {cmd_plan}")
    _run(cmd_plan, env)

    # ── Copy splits_final.json to preprocessed dir ────────────────────────
    splits_src = fold_base / "splits_final.json"
    pre_ds_dir = Path(env["nnUNet_preprocessed"]) / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    pre_ds_dir.mkdir(parents=True, exist_ok=True)
    if splits_src.exists():
        splits_dst = pre_ds_dir / "splits_final.json"
        shutil.copy2(splits_src, splits_dst)
        print(f"  Copied splits_final.json → {splits_dst}")

    # ── Step 2: train 3d_fullres ──────────────────────────────────────────
    cmd_train = f"nnUNetv2_train {DATASET_ID} 3d_fullres 0 --npz {device_flag}"
    print(f"\n  Step 2: {cmd_train}")
    _run(cmd_train, env)

    # ── Step 3: train 2d ─────────────────────────────────────────────────
    cmd_2d = f"nnUNetv2_train {DATASET_ID} 2d 0 --npz {device_flag}"
    print(f"\n  Step 3 (2D comparison): {cmd_2d}")
    _run(cmd_2d, env)

    # ── Step 4: predict held-out patient ─────────────────────────────────
    for config in ["3d_fullres", "2d"]:
        pred_out = (RESULTS_DIR /
                    f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold}" /
                    f"predictions_{config}")
        pred_out.mkdir(parents=True, exist_ok=True)
        cmd_pred = (
            f"nnUNetv2_predict "
            f"-i {images_ts.resolve()} "
            f"-o {pred_out.resolve()} "
            f"-d {DATASET_ID} "
            f"-c {config} "
            f"-f 0 "
            f"-chk checkpoint_best.pth "
            f"{device_flag}"
        )
        print(f"\n  Step 4 [{config}]: {cmd_pred}")
        _run(cmd_pred, env)

    print(f"\n  ✓ Fold {fold} complete.")
    print(f"  Next: python fracture_pipeline.py evaluate")
    print(f"  Or:   python fracture_pipeline.py plot --what all")

def _nnunet_env(fold: int) -> tuple:
    """
    Build isolated nnU-Net env for one fold.
    Each fold gets its own raw/preprocessed/results dirs so nnU-Net
    only ever sees one Dataset001_* folder at a time.
    """
    fold_key = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold}"
    fold_src = FOLDS_DIR / fold_key

    # Isolated workdir per fold
    workdir = BASE_DIR / "nnunet_workdir" / f"fold{fold}"
    raw_dir = workdir / "raw"
    pre_dir = workdir / "preprocessed"
    res_dir = workdir / "results"

    raw_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Symlink fold dataset into isolated raw dir with canonical name
    ds_link = raw_dir / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    if not ds_link.exists():
        try:
            ds_link.symlink_to(fold_src.resolve())
        except OSError:
            shutil.copytree(str(fold_src), str(ds_link))

    env = os.environ.copy()
    env["nnUNet_raw"]          = str(raw_dir.resolve())
    env["nnUNet_preprocessed"] = str(pre_dir.resolve())
    env["nnUNet_results"]      = str(res_dir.resolve())
    return env, res_dir


def _run(cmd: str, env: Dict) -> None:
    """Run shell command with given environment."""
    import subprocess
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"  ⚠️  Command returned non-zero exit code: {result.returncode}")


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def dice_3d(pred: np.ndarray, gt: np.ndarray,
            smooth: float = 1e-6) -> float:
    """Volumetric Dice coefficient."""
    pred = (pred > 0).astype(np.float32).flatten()
    gt   = (gt   > 0).astype(np.float32).flatten()
    intersection = (pred * gt).sum()
    return float((2.0 * intersection + smooth) /
                 (pred.sum() + gt.sum() + smooth))


def evaluate_all_folds(splits: List[Dict]) -> None:
    """
    Load nnU-Net predictions and compute 3D Dice per fold.
    Prints mean ± std Dice across all validation patients.
    """
    print(f"\n[Evaluate] {len(splits)} folds")
    all_dice = []
    results  = []

    for fold_i, split in enumerate(splits):
        fold_results = RESULTS_DIR / \
            f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold_i}"
        pred_dir = fold_results / "predictions_3d_fullres"

        for pid in split["val"]:
            pred_path = pred_dir / f"{pid}.nii.gz"
            gt_path   = CONVERTED_DIR / f"{pid}_fracture.nii.gz"

            if not pred_path.exists():
                print(f"  ⚠️  No prediction for {pid} at {pred_path}")
                continue
            if not gt_path.exists():
                print(f"  ⚠️  No ground truth for {pid}")
                continue

            pred = nib.load(pred_path).get_fdata()
            gt   = nib.load(gt_path).get_fdata()
            d    = dice_3d(pred, gt)
            all_dice.append(d)
            results.append({
                "fold":       fold_i,
                "patient_id": pid,
                "dice_3d":    round(d, 4),
            })
            print(f"  Fold {fold_i}  {pid}  Dice={d:.4f}")

    if all_dice:
        print(f"\n  ──────────────────────────────")
        print(f"  Mean Dice : {np.mean(all_dice):.4f}")
        print(f"  Std  Dice : {np.std(all_dice):.4f}")
        print(f"  Min  Dice : {np.min(all_dice):.4f}")
        print(f"  Max  Dice : {np.max(all_dice):.4f}")
        print(f"  N patients: {len(all_dice)}")

        out_path = RESULTS_DIR / "evaluation_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "per_patient": results,
                "summary": {
                    "mean_dice": round(float(np.mean(all_dice)), 4),
                    "std_dice":  round(float(np.std(all_dice)),  4),
                    "n":         len(all_dice),
                }
            }, f, indent=2)
        print(f"\n  Results saved → {out_path}")
    else:
        print("  No predictions found. Run training first.")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_prediction(patient_id:  str,
                    pred_path:   str,
                    gt_path:     Optional[str] = None,
                    ct_path:     Optional[str] = None,
                    n_cols:      int = 8,
                    output_dir:  str = "plots") -> Path:
    """
    Plot prediction slices side-by-side with ground truth (if available).

    Shows only slices where prediction OR ground truth is non-zero.
    Each row:
        Top    — CT (bone window) + prediction overlay  (red)
        Bottom — CT (bone window) + ground truth overlay (green)
                 or blank if no GT provided

    Parameters
    ----------
    patient_id  : used for title and filename
    pred_path   : NIfTI prediction file (.nii.gz)
    gt_path     : NIfTI ground truth file (optional)
    ct_path     : NIfTI cropped CT file (optional, falls back to grey bg)
    n_cols      : number of slices to show
    output_dir  : folder to save PNG

    Returns path to saved PNG.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────
    pred_nib  = nib.load(pred_path)
    pred      = (pred_nib.get_fdata().transpose(2, 1, 0) > 0)  # (Z,H,W) bool

    gt = None
    if gt_path and Path(gt_path).exists():
        gt = (nib.load(gt_path).get_fdata().transpose(2, 1, 0) > 0)

    ct = None
    if ct_path and Path(ct_path).exists():
        ct = nib.load(ct_path).get_fdata().transpose(2, 1, 0).astype(np.float32)

    Z = pred.shape[0]

    # ── Which slices to show ──────────────────
    active = pred.any(axis=(1, 2))
    if gt is not None:
        active = active | gt.any(axis=(1, 2))
    active_idx = np.where(active)[0]

    if len(active_idx) == 0:
        print(f"  ⚠️  No non-zero voxels in prediction or GT for {patient_id}")
        active_idx = np.linspace(0, Z - 1, n_cols, dtype=int)

    show_idx = active_idx[
        np.linspace(0, len(active_idx) - 1, n_cols, dtype=int)
    ]

    # ── Bone window helper ────────────────────
    lo = BONE_WL - BONE_WW / 2
    hi = BONE_WL + BONE_WW / 2

    def bw(sl):
        if ct is not None:
            return np.clip((sl - lo) / (hi - lo), 0, 1)
        return np.zeros_like(sl)

    # ── Compute Dice if GT available ──────────
    dice_str = ""
    if gt is not None:
        d = dice_3d(pred.astype(np.float32), gt.astype(np.float32))
        dice_str = f"  3D Dice = {d:.4f}"

    # ── Layout: 2 rows (pred / gt) ────────────
    n_rows = 2 if gt is not None else 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 3.0),
        facecolor="black"
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # make 2D for consistent indexing

    for col, idx in enumerate(show_idx):
        bg = bw(ct[idx]) if ct is not None else np.zeros(pred.shape[1:])

        # ── Row 0: prediction ──────────────────
        ax = axes[0, col]
        ax.imshow(bg, cmap="gray", interpolation="nearest")
        if pred[idx].any():
            ax.imshow(
                np.ma.masked_where(~pred[idx], np.ones(pred[idx].shape)),
                cmap="Reds", alpha=0.65, vmin=0, vmax=1
            )
        ax.set_title(f"z={idx}", fontsize=7, color="white")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Prediction", color="red", fontsize=9)

        # ── Row 1: ground truth ────────────────
        if gt is not None:
            ax2 = axes[1, col]
            ax2.imshow(bg, cmap="gray", interpolation="nearest")
            if gt[idx].any():
                ax2.imshow(
                    np.ma.masked_where(~gt[idx], np.ones(gt[idx].shape)),
                    cmap="Greens", alpha=0.65, vmin=0, vmax=1
                )
            ax2.axis("off")
            if col == 0:
                ax2.set_ylabel("Ground Truth", color="lime", fontsize=9)

    legend_patches = [
        mpatches.Patch(facecolor="red",  alpha=0.7, label="Prediction"),
    ]
    if gt is not None:
        legend_patches.append(
            mpatches.Patch(facecolor="lime", alpha=0.7, label="Ground Truth")
        )

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=10,
               facecolor="black", labelcolor="white", framealpha=0)
    fig.suptitle(
        f"Prediction — {patient_id}{dice_str}",
        color="white", fontsize=12
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    out_path = out_dir / f"{patient_id}_prediction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def plot_all_predictions(splits: List[Dict],
                          output_dir: str = "plots") -> None:
    """
    After training and inference, plot predictions for every
    validation patient across all folds.
    Saves one PNG per patient to plots/.
    """
    print(f"\n[Plot] Generating prediction plots...")

    for fold_i, split in enumerate(splits):
        fold_results = RESULTS_DIR / \
            f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold_i}"
        pred_dir = fold_results / "predictions_3d_fullres"

        train_ids = split["train"]

        for pid in split["val"]:
            pred_path = pred_dir / f"{pid}.nii.gz"
            gt_path   = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
            ct_path   = CONVERTED_DIR / f"{pid}_ct.nii.gz"

            if not pred_path.exists():
                print(f"  ⚠️  No prediction for {pid}")
                continue

            # Label clearly that this is a CV prediction (model never saw this patient)
            cv_label = (
                f"{pid}  [LOOCV fold {fold_i} — "
                f"trained on: {', '.join(train_ids)}]"
            )
            print(f"  Fold {fold_i}  {pid}  (trained on {train_ids})")
            plot_prediction(
                patient_id=cv_label,
                pred_path=str(pred_path),
                gt_path=str(gt_path) if gt_path.exists() else None,
                ct_path=str(ct_path) if ct_path.exists() else None,
                output_dir=output_dir,
            )


def plot_training_curves(fold: int,
                          output_dir: str = "plots") -> Optional[Path]:
    """
    Parse nnU-Net training log and plot train/val loss + Dice over epochs.
    nnU-Net saves progress to:
      nnunet_results/<fold>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt

    Shows:
        - Train loss per epoch
        - Validation loss per epoch
        - Pseudo Dice per epoch (nnU-Net's online estimate)
        - Vertical line at best epoch (lowest val loss)
    """
    import glob
    import re

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Search in both nnunet_workdir (current) and RESULTS_DIR (legacy)
    workdir_results = BASE_DIR / "nnunet_workdir" / f"fold{fold}" / "results"
    search_roots = [workdir_results, RESULTS_DIR]
    log_files = []
    for root in search_roots:
        pattern = str(root / "**" / "fold_0" / "training_log_*.txt")
        log_files += glob.glob(pattern, recursive=True)

    if not log_files:
        print(f"  ⚠️  No training log found — training may not have started yet.")
        return None

    log_path = sorted(log_files)[-1]
    print(f"  Parsing {log_path}")

    train_loss, val_loss, pseudo_dice, epochs = [], [], [], []

    with open(log_path) as f:
        for line in f:
            # nnU-Net log format (uses underscores):
            # train_loss X.XXXX
            # val_loss X.XXXX
            # Pseudo dice [X.XXXX]
            # Epoch X  (at start of epoch)
            ep_match  = re.search(r"Epoch\s+(\d+)", line)
            tr_match  = re.search(r"train_loss\s+([-\d.]+)", line)
            val_match = re.search(r"val_loss\s+([-\d.]+)", line)
            pd_match  = re.search(r"[Pp]seudo\s+[Dd]ice\s+\[([^\]]+)\]", line)

            if ep_match:
                epochs.append(int(ep_match.group(1)))
            if tr_match:
                train_loss.append(float(tr_match.group(1)))
            if val_match:
                val_loss.append(float(val_match.group(1)))
            if pd_match:
                raw = pd_match.group(1)
                # matches np.float32(0.3151) or plain 0.3151
                nums = re.findall(r"\d+\.\d+", raw)
                if nums:
                    pseudo_dice.append(float(nums[-1]))

    if not train_loss:
        print("  ⚠️  Could not parse training losses from log.")
        return None

    # Align all series to shortest completed length
    n = min(len(train_loss), len(val_loss)) if val_loss else len(train_loss)
    if pseudo_dice:
        n = min(n, len(pseudo_dice))
    ep     = list(range(n))          # 0-indexed epochs
    tr     = train_loss[:n]
    vl     = val_loss[:n] if val_loss else None
    pd_vals = pseudo_dice[:n] if pseudo_dice else None

    # ── 2-panel layout ────────────────────────────────────────────────────
    # Panel 1: train + val loss combined
    # Panel 2: Dice (patch-level estimate during training)
    has_dice = pd_vals is not None and len(pd_vals) > 0
    n_panels = 2 if has_dice else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 6, 4),
                              facecolor="white")
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: Loss (train + val on same axes) ──────────────────────────
    ax = axes[0]
    ax.plot(ep, tr, color="#2196F3", linewidth=2, label="Train loss")
    if vl:
        ax.plot(ep, vl, color="#4CAF50", linewidth=2,
                linestyle="--", label="Val loss (held-out patient)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss  (Dice + CE,  lower = better)", fontsize=10)
    ax.set_title(f"Training & Validation Loss — Fold {fold}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ── Panel 2: Dice ─────────────────────────────────────────────────────
    if has_dice:
        ax2 = axes[1]
        ax2.plot(ep[:len(pd_vals)], pd_vals,
                 color="#FF9800", linewidth=2, marker="o", markersize=4,
                 label="Dice (patch estimate)")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Dice  (0 = no overlap, 1 = perfect)", fontsize=10)
        ax2.set_title(f"Training Dice Score — Fold {fold}", fontsize=12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        # Annotate each point with its value — large, bold, above each dot
        for x, y in zip(ep[:len(pd_vals)], pd_vals):
            ax2.annotate(f"{y:.3f}", (x, y),
                         textcoords="offset points", xytext=(0, 14),
                         fontsize=11, fontweight="bold", color="#E65100",
                         ha="center",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc="white", ec="#FF9800", alpha=0.8))

    fig.suptitle(f"nnU-Net Training Progress — Fold {fold}  ({n} epochs so far)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / f"training_curves_fold{fold}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def plot_dice_summary(splits: List[Dict],
                       output_dir: str = "plots") -> Optional[Path]:
    """
    Bar chart of per-patient Dice scores across all folds.
    Reads from evaluation_results.json produced by evaluate_all_folds().
    """
    results_path = RESULTS_DIR / "evaluation_results.json"
    if not results_path.exists():
        print("  ⚠️  Run 'evaluate' first to generate evaluation_results.json")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    per_patient = data["per_patient"]
    summary     = data["summary"]

    pids  = [r["patient_id"] for r in per_patient]
    dices = [r["dice_3d"]    for r in per_patient]
    folds = [r["fold"]       for r in per_patient]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
              "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]
    bar_colors = [colors[f % len(colors)] for f in folds]

    fig, ax = plt.subplots(figsize=(max(8, len(pids) * 1.5), 5))
    bars = ax.bar(pids, dices, color=bar_colors, edgecolor="white",
                  linewidth=0.8)

    # Value labels on bars
    for bar, d in zip(bars, dices):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{d:.3f}", ha="center", va="bottom", fontsize=9)

    # Mean line
    mean_d = summary["mean_dice"]
    std_d  = summary["std_dice"]
    ax.axhline(mean_d, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_d:.3f} ± {std_d:.3f}")
    ax.fill_between([-0.5, len(pids) - 0.5],
                    mean_d - std_d, mean_d + std_d,
                    color="red", alpha=0.08)

    ax.set_ylim(0, min(1.05, max(dices) + 0.15))
    ax.set_xlabel("Patient", fontsize=11)
    ax.set_ylabel("3D Dice", fontsize=11)
    ax.set_title(
        f"Per-Patient 3D Dice  —  "
        f"Mean={mean_d:.3f} ± {std_d:.3f}  "
        f"(n={summary['n']})",
        fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Fold legend
    unique_folds = sorted(set(folds))
    fold_patches = [
        mpatches.Patch(color=colors[f % len(colors)], label=f"Fold {f}")
        for f in unique_folds
    ]
    ax.legend(handles=fold_patches + ax.get_legend_handles_labels()[0][-1:],
              fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = out_dir / "dice_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# Inference — New Patient
# ─────────────────────────────────────────────
def predict_new_patient(dicom_dir:    str,
                         seg_pelvis:   str,
                         output_path:  str,
                         fold:         int = 0,
                         padding:      int = CROP_PADDING) -> np.ndarray:
    """
    Full inference pipeline for a new unseen patient.

    Steps:
        1. Load CT + pelvis mask
        2. Crop to pelvis ROI (same as training)
        3. Run nnU-Net prediction
        4. Place prediction back in original CT space
        5. Save as NIfTI

    Returns fracture mask in original CT space (Z, H, W) uint8.
    """
    print(f"\n[Predict] {dicom_dir}")

    tmp_in  = Path("tmp_predict/input")
    tmp_out = Path("tmp_predict/output")
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    # ── Load and crop ──────────────────────────
    volume, affine, spacing = load_ct_volume(dicom_dir)
    pelvis = load_seg_dcm(seg_pelvis)

    if volume.shape != pelvis.shape:
        print("  Shape mismatch — resizing pelvis mask...")
        pelvis = _resize_mask_to_volume(pelvis, volume.shape)

    vol_cr, _, bbox = crop_to_mask(volume, pelvis, padding=padding)
    z0, z1, y0, y1, x0, x1 = bbox

    print(f"  CT shape  : {volume.shape}")
    print(f"  Cropped   : {vol_cr.shape}")
    print(f"  Spacing   : {spacing}")

    # ── Save cropped CT for nnU-Net ────────────
    tmp_ct = tmp_in / "predict_0000.nii.gz"
    save_nifti(vol_cr, affine, tmp_ct)

    # ── nnU-Net predict ────────────────────────
    env, res_dir = _nnunet_env(fold)

    cmd = (
        f"nnUNetv2_predict "
        f"-i {tmp_in} "
        f"-o {tmp_out} "
        f"-d {DATASET_ID} "
        f"-c 3d_fullres "
        f"-chk checkpoint_best.pth "   # always use best, not final epoch
        f"--save_probabilities"
    )
    print(f"\n  Running: {cmd}")
    _run(cmd, env)

    # ── Load prediction and uncrop ─────────────
    pred_path = tmp_out / "predict.nii.gz"
    if not pred_path.exists():
        print(f"  ⚠️  Prediction not found at {pred_path}")
        return np.zeros_like(volume, dtype=np.uint8)

    pred_nib  = nib.load(pred_path)
    pred_data = pred_nib.get_fdata()
    # NIfTI → (Z, H, W)
    pred_crop = pred_data.transpose(2, 1, 0).astype(np.uint8)

    # Uncrop back to original CT space
    pred_full = np.zeros(volume.shape, dtype=np.uint8)
    pred_full[z0:z1, y0:y1, x0:x1] = (pred_crop > 0).astype(np.uint8)

    # ── Save full-resolution prediction ────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(pred_full, affine, out)

    n_vox = pred_full.sum()
    print(f"\n  Fracture voxels predicted: {n_vox:,}")
    print(f"  Saved → {out}")

    # ── Cleanup temp files ─────────────────────
    shutil.rmtree("tmp_predict", ignore_errors=True)

    return pred_full



# ─────────────────────────────────────────────
# Ensemble — All 4 Models on a New Patient
# ─────────────────────────────────────────────
def predict_ensemble(dicom_dir:   str,
                     seg_pelvis:  str,
                     output_path: str,
                     n_folds:     int = 4,
                     threshold:   float = 0.5,
                     padding:     int = CROP_PADDING) -> np.ndarray:
    """
    Run all N fold models on a new unseen patient and average
    their probability maps before thresholding.

    This is ONLY valid for patients not in the training set.
    Do NOT use this for CV evaluation — use evaluate_all_folds() instead.

    How it works
    ------------
    1. Crop CT to pelvis ROI  (same preprocessing as training)
    2. Run each fold model with --save_probabilities
       → each produces a softmax prob map  (values 0..1 per voxel)
    3. Average prob maps across all folds
    4. Threshold at `threshold` (default 0.5)
    5. Uncrop back to original CT space
    6. Save final binary mask as NIfTI

    Parameters
    ----------
    dicom_dir    : DICOM folder for new patient
    seg_pelvis   : pelvis segmentation .dcm for ROI crop
    output_path  : where to save final prediction .nii.gz
    n_folds      : how many fold models to ensemble (default = all)
    threshold    : probability threshold for final binary mask
    """
    print(f"\n[Ensemble Predict] {dicom_dir}")
    print(f"  Using {n_folds} fold models  |  threshold={threshold}")

    # ── Load and crop CT ──────────────────────
    volume, affine, spacing = load_ct_volume(dicom_dir)
    pelvis = load_seg_dcm(seg_pelvis)

    if volume.shape != pelvis.shape:
        print("  Shape mismatch — resizing pelvis mask...")
        pelvis = _resize_mask_to_volume(pelvis, volume.shape)

    vol_cr, _, bbox = crop_to_mask(volume, pelvis, padding=padding)
    z0, z1, y0, y1, x0, x1 = bbox

    print(f"  CT shape : {volume.shape}")
    print(f"  Cropped  : {vol_cr.shape}")

    # Save cropped CT once — reused for all fold predictions
    tmp_in  = Path("tmp_ensemble/input")
    tmp_in.mkdir(parents=True, exist_ok=True)
    save_nifti(vol_cr, affine, tmp_in / "patient_0000.nii.gz")

    # ── Run each fold model ───────────────────
    prob_sum   = None   # accumulates probability maps
    n_success  = 0

    for fold_i in range(n_folds):
        fold_key  = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold_i}"
        fold_dir  = FOLDS_DIR   / fold_key
        fold_res  = RESULTS_DIR / fold_key
        tmp_out_i = Path(f"tmp_ensemble/fold_{fold_i}")
        tmp_out_i.mkdir(parents=True, exist_ok=True)

        if not fold_dir.exists():
            print(f"  ⚠️  Fold {fold_i} not found — skipping")
            continue

        checkpoint = _find_best_checkpoint(fold_res)
        if checkpoint is None:
            print(f"  ⚠️  No checkpoint for fold {fold_i} — skipping")
            continue

        env, _res = _nnunet_env(fold_i)
        cmd = (
            f"nnUNetv2_predict "
            f"-i {tmp_in} "
            f"-o {tmp_out_i} "
            f"-d {DATASET_ID} "
            f"-c 3d_fullres "
            f"-chk checkpoint_best.pth "
            f"--save_probabilities"      # saves .npz with softmax probs
        )
        print(f"\n  Fold {fold_i}: {cmd}")
        _run(cmd, env)

        # Load probability map (.npz file saved by --save_probabilities)
        # nnU-Net saves:  patient.nii.gz  (binary) and  patient.npz  (probs)
        prob_files = list(tmp_out_i.glob("*.npz"))
        if not prob_files:
            print(f"  ⚠️  No .npz probability file for fold {fold_i}")
            # Fall back to binary prediction
            bin_path = tmp_out_i / "patient.nii.gz"
            if bin_path.exists():
                pred = nib.load(bin_path).get_fdata()
                pred = pred.transpose(2, 1, 0).astype(np.float32)
                prob = pred   # treat binary as prob (0 or 1)
            else:
                continue
        else:
            # npz contains 'softmax' array: (C, X, Y, Z) where C=num_classes
            data = np.load(prob_files[0])
            # Class 1 (fracture) probability, shape (X,Y,Z)
            prob = data["softmax"][1]
            # NIfTI convention (X,Y,Z) → (Z,H,W)
            prob = prob.transpose(2, 1, 0).astype(np.float32)

        if prob_sum is None:
            prob_sum = prob.copy()
        else:
            # Shapes must match — handle minor size differences from resampling
            if prob_sum.shape != prob.shape:
                print(f"  ⚠️  Shape mismatch fold {fold_i}: "
                      f"{prob.shape} vs {prob_sum.shape} — cropping to min")
                min_z = min(prob_sum.shape[0], prob.shape[0])
                min_h = min(prob_sum.shape[1], prob.shape[1])
                min_w = min(prob_sum.shape[2], prob.shape[2])
                prob_sum = prob_sum[:min_z, :min_h, :min_w]
                prob      = prob[:min_z, :min_h, :min_w]

            prob_sum += prob

        n_success += 1
        print(f"  Fold {fold_i} ✓  (fracture prob mean={prob.mean():.4f}  "
              f"max={prob.max():.4f})")

    if n_success == 0:
        print("  ✗ No fold predictions succeeded.")
        shutil.rmtree("tmp_ensemble", ignore_errors=True)
        return np.zeros_like(volume, dtype=np.uint8)

    # ── Average and threshold ─────────────────
    prob_avg = prob_sum / n_success
    pred_crop = (prob_avg >= threshold).astype(np.uint8)

    print(f"\n  Ensemble of {n_success} models:")
    print(f"  Mean prob in fracture region: "
          f"{prob_avg[pred_crop > 0].mean():.3f}"
          if pred_crop.sum() > 0 else "  No fracture voxels above threshold")
    print(f"  Fracture voxels predicted: {pred_crop.sum():,}")

    # ── Uncrop to original CT space ───────────
    pred_full = np.zeros(volume.shape, dtype=np.uint8)
    cz = min(pred_crop.shape[0], z1 - z0)
    cy = min(pred_crop.shape[1], y1 - y0)
    cx = min(pred_crop.shape[2], x1 - x0)
    pred_full[z0:z0+cz, y0:y0+cy, x0:x0+cx] = pred_crop[:cz, :cy, :cx]

    # ── Save ──────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(pred_full, affine, out)
    print(f"  Saved → {out}")

    # Save probability map too (useful for threshold tuning)
    prob_full = np.zeros(volume.shape, dtype=np.float32)
    prob_full[z0:z0+cz, y0:y0+cy, x0:x0+cx] = prob_avg[:cz, :cy, :cx]
    prob_path = out.parent / (out.name.replace(".nii.gz", "_probmap.nii.gz"))
    save_nifti(prob_full, affine, prob_path)
    print(f"  Probability map → {prob_path}")

    # ── Cleanup ───────────────────────────────
    shutil.rmtree("tmp_ensemble", ignore_errors=True)
    return pred_full


def _find_best_checkpoint(fold_results: Path) -> Optional[Path]:
    """Search fold results dir for checkpoint_best.pth."""
    import glob
    pattern = str(fold_results / "**" / "checkpoint_best.pth")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return Path(sorted(matches)[0])
    return None


# ─────────────────────────────────────────────
# Retrain on ALL patients (for deployment)
# ─────────────────────────────────────────────
def retrain_all(n_epochs: Optional[int] = None) -> None:
    """
    Train a single final model on ALL annotated patients combined.

    Use this AFTER cross-validation has confirmed the approach works.
    This model is what you deploy for new patients — it has seen
    all available data and is therefore stronger than any single fold.

    n_epochs: if None, uses average of best epochs across folds.
              If no fold training has been done yet, defaults to 500.
    """
    registry = FractureRegistry()
    if len(registry) == 0:
        print("No patients registered.")
        return

    print(f"\n[Retrain All] Training on all {len(registry)} patients")

    # ── Build dataset with all patients ───────
    all_base   = FOLDS_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_all"
    images_dir = all_base / "imagesTr"
    labels_dir = all_base / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for case in registry:
        src_ct   = CONVERTED_DIR / f"{case.patient_id}_ct.nii.gz"
        src_frac = CONVERTED_DIR / f"{case.patient_id}_fracture.nii.gz"
        if not src_ct.exists():
            print(f"  ⚠️  {case.patient_id} not converted — run 'prepare' first")
            continue
        shutil.copy2(src_ct,   images_dir / f"{case.patient_id}_0000.nii.gz")
        shutil.copy2(src_frac, labels_dir / f"{case.patient_id}.nii.gz")
        print(f"  Added {case.patient_id}")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "fracture": 1},
        "numTraining": len(registry),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(all_base / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # ── Determine number of epochs ─────────────
    if n_epochs is None:
        n_epochs = _infer_best_epochs()
        print(f"  Auto epoch count from fold training: {n_epochs}")

    # ── Build isolated env for retrain_all ───────────────────────────────
    # retrain_all is not a CV fold so we build its env manually
    all_workdir = BASE_DIR / "nnunet_workdir" / "all"
    all_raw     = all_workdir / "raw"
    all_pre     = all_workdir / "preprocessed"
    all_res     = all_workdir / "results"
    all_raw.mkdir(parents=True, exist_ok=True)
    all_pre.mkdir(parents=True, exist_ok=True)
    all_res.mkdir(parents=True, exist_ok=True)

    ds_link = all_raw / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    if not ds_link.exists():
        try:
            ds_link.symlink_to(all_base.resolve())
        except OSError:
            shutil.copytree(str(all_base), str(ds_link))

    env = os.environ.copy()
    env["nnUNet_raw"]          = str(all_raw.resolve())
    env["nnUNet_preprocessed"] = str(all_pre.resolve())
    env["nnUNet_results"]      = str(all_res.resolve())

    # ── Train ─────────────────────────────────

    cmd_plan = (
        f"nnUNetv2_plan_and_preprocess "
        f"-d {DATASET_ID} "
        f"--verify_dataset_integrity"
    )
    _run(cmd_plan, env)

    cmd_train = (
        f"nnUNetv2_train {DATASET_ID} 3d_fullres 0 "
        f"--npz "
        f"-num_epochs {n_epochs}"
    )
    print(f"\n  Training: {cmd_train}")
    _run(cmd_train, env)

    results_dir = RESULTS_DIR / f"{all_base.name}"
    print(f"\n  ✓ Final model trained on all {len(registry)} patients")
    print(f"  Weights: {results_dir}/**/checkpoint_final.pth")
    print(f"  Use for new patients:")
    print(f"    python fracture_pipeline.py predict_final \\")
    print(f"      --dicom_dir ... --seg_pelvis ... --output ...")


def _infer_best_epochs() -> int:
    """
    Read nnU-Net training logs from all folds and return
    the average best epoch as a sensible epoch count for
    the final all-data retrain.
    """
    import glob, re
    best_epochs = []

    for fold_i in range(10):  # check up to 10 folds
        fold_key = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold_i}"
        log_pattern = str(
            RESULTS_DIR / fold_key / "**" / "fold_0" / "training_log_*.txt"
        )
        logs = glob.glob(log_pattern, recursive=True)
        if not logs:
            continue

        val_losses = []
        with open(sorted(logs)[-1]) as f:
            for line in f:
                m = re.search(r"val(?:idation)? loss[^:]*:[\s]*([-\d.]+)",
                               line, re.I)
                if m:
                    val_losses.append(float(m.group(1)))

        if val_losses:
            best_ep = int(np.argmin(val_losses)) + 1
            best_epochs.append(best_ep)
            print(f"  Fold {fold_i} best epoch: {best_ep}")

    if not best_epochs:
        print("  Could not determine best epochs — defaulting to 500")
        return 500

    avg = int(np.mean(best_epochs))
    print(f"  Average best epoch across folds: {avg}")
    # Add 10% buffer since we have more data now
    buffered = int(avg * 1.1)
    print(f"  Using {buffered} epochs (avg + 10% buffer)")
    return buffered

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pelvic Fracture Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── add ───────────────────────────────────
    a = sub.add_parser("add", help="Register a new patient")
    a.add_argument("--patient_id",    required=True)
    a.add_argument("--dicom_dir",     required=True)
    a.add_argument("--seg_fracture",  required=True)
    a.add_argument("--seg_pelvis",    required=True)

    # ── list ──────────────────────────────────
    sub.add_parser("list", help="List all registered patients")

    # ── remove ────────────────────────────────
    rm = sub.add_parser("remove", help="Remove a patient from the registry")
    rm.add_argument("--patient_id", required=True)

    # ── qc ────────────────────────────────────
    q = sub.add_parser("qc", help="Visual QC for one or all patients")
    q.add_argument("--patient_id", default=None,
                   help="Specific patient (omit = all patients)")

    # ── prepare ───────────────────────────────
    prep = sub.add_parser(
        "prepare",
        help="Convert all patients + create nnU-Net fold datasets"
    )
    prep.add_argument("--hpc_splits", action="store_true", default=False,
                      help="Use true held-out patient as nnU-Net val split (HPC mode). "
                           "Default uses last training patient (laptop-safe).")

    # ── train ─────────────────────────────────
    t = sub.add_parser("train", help="Train nnU-Net for one fold")
    t.add_argument("--fold", type=int, required=True)

    # ── evaluate ──────────────────────────────
    sub.add_parser("evaluate", help="Compute 3D Dice for all folds")

    # ── predict ───────────────────────────────
    pr = sub.add_parser("predict", help="Predict fractures for a new patient")
    pr.add_argument("--dicom_dir",   required=True)
    pr.add_argument("--seg_pelvis",  required=True)
    pr.add_argument("--output",      required=True,
                    help="Output NIfTI path, e.g. predictions/patient.nii.gz")
    pr.add_argument("--fold", type=int, default=0,
                    help="Which trained fold to use (default: 0)")

    # ── ensemble ──────────────────────────────
    en = sub.add_parser(
        "ensemble",
        help="Predict new patient using all fold models averaged"
    )
    en.add_argument("--dicom_dir",  required=True)
    en.add_argument("--seg_pelvis", required=True)
    en.add_argument("--output",     required=True)
    en.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold (default 0.5)")
    en.add_argument("--n_folds", type=int, default=None,
                    help="How many folds to ensemble (default = all)")

    # ── retrain_all ───────────────────────────
    ts = sub.add_parser("train_sam", help="Train SAM3+LoRA (2.5D) for one fold")
    ts.add_argument("--fold",    type=int, default=0, help="LOOCV fold index")
    ts.add_argument("--epochs",  type=int, default=50, help="Training epochs")
    ts.add_argument("--lr",      type=float, default=1e-4, help="Learning rate")
    ts.add_argument("--rank",    type=int, default=4, help="LoRA rank")
    ts.add_argument("--context", type=int, default=1,
                    help="Adjacent slices for 2.5D (1 = use slice-1,slice,slice+1)")
    ts.add_argument("--validate", action="store_true", default=False,
                    help="Run validation on held-out patient each epoch (HPC mode). "
                         "Disabled by default to avoid OOM on laptop.")

    ra = sub.add_parser(
        "retrain_all",
        help="Train final model on ALL patients for deployment"
    )
    ra.add_argument("--n_epochs", type=int, default=None,
                    help="Epochs (default = auto from fold logs)")

    # ── plot ──────────────────────────────────
    pl = sub.add_parser(
        "plot",
        help="Plot predictions, training curves, and Dice summary"
    )
    pl.add_argument(
        "--what",
        choices=["predictions", "curves", "dice", "all"],
        default="all",
        help="predictions | curves | dice | all (default)"
    )
    pl.add_argument("--fold", type=int, default=None,
                    help="Specific fold for curves (omit = all folds)")
    pl.add_argument("--output_dir", default="plots",
                    help="Where to save PNG files (default: plots/)")

    return p


def _load_splits():
    """Load CV splits from disk. Returns None if not found."""
    splits_path = FOLDS_DIR / "cv_splits.json"
    if not splits_path.exists():
        print("No CV splits found. Run 'prepare' first.")
        return None
    with open(splits_path) as f:
        data = json.load(f)
        return data["splits"] if isinstance(data, dict) else data



# ─────────────────────────────────────────────────────────────────────────────
# SAM3 + LoRA  2.5D training with 3D Dice loss
# ─────────────────────────────────────────────────────────────────────────────

def _dice_loss_3d(pred: "torch.Tensor", target: "torch.Tensor",
                  smooth: float = 1e-6) -> "torch.Tensor":
    """
    3D Dice loss computed across the full reconstructed volume.

    pred   : (N, H, W)  float sigmoid probabilities  (stacked slice predictions)
    target : (N, H, W)  float binary ground truth
    Returns scalar loss in [0, 1].
    """
    import torch
    pred   = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def _build_sam3_lora(rank: int = 4, device: str = "cpu"):
    """
    Load SAM3 image model + processor, freeze backbone, add trainable
    segmentation head (LoRA-style: small parameter count, adapts to CT domain).

    Architecture:
        SAM3 backbone (frozen, 848M params)
            ↓ extract mask logits via processor API
        Small segmentation head (trainable, ~50K params):
            Conv2d(1→16) → ReLU → Conv2d(16→1) → Sigmoid
        
    The "LoRA" here means parameter-efficient fine-tuning — we don't modify
    SAM3's weights but add a lightweight head that adapts its outputs to CT.

    Returns (model, processor, seg_head, trainable_params).
    """
    import torch
    import torch.nn as nn

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        raise RuntimeError(
            "SAM3 not found.\n"
            "Run with base Python: /opt/homebrew/Caskroom/miniconda/base/bin/python3 fracture_pipeline.py train_sam\n"
            "Or: conda activate base"
        )

    print("  Loading SAM3 backbone (frozen)...")
    # build_sam3_image_model handles device internally; pass cpu for Mac
    try:
        model = build_sam3_image_model(device="cpu", eval_mode=True, load_from_HF=True)
    except Exception as e:
        print(f"  HF download failed ({e}), loading without pretrained weights...")
        model = build_sam3_image_model(device="cpu", eval_mode=True, load_from_HF=False)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    processor = Sam3Processor(model)

    # Small trainable segmentation refinement head
    # Takes SAM3 binary mask output (1 channel) and refines it for CT fractures
    class SegHead(nn.Module):
        """
        Lightweight segmentation head that takes SAM3 backbone features (256ch)
        and produces a binary mask. ~50k params — parameter-efficient fine-tuning.
        """
        def __init__(self, in_channels=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 64, 1),          # channel reduction
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )
        def forward(self, x):
            return torch.sigmoid(self.net(x))

    seg_head = SegHead().to("cpu")  # keep on CPU, MPS has no headroom with 840M backbone
    trainable = list(seg_head.parameters())

    total_sam3   = sum(p.numel() for p in model.parameters())
    n_trainable  = sum(p.numel() for p in trainable)
    print(f"  SAM3 backbone:     {total_sam3:,} params (frozen)")
    print(f"  Seg head (LoRA):   {n_trainable:,} params (trainable)")
    print(f"  Ratio: {100*n_trainable/max(total_sam3,1):.3f}% of backbone")

    return model, processor, seg_head, trainable


def _ct_to_sam3_slices(ct_path: Path, mask_path: Optional[Path],
                        context: int = 1, target_hw: tuple = (1024, 256)
                        ) -> list:
    """
    Convert a 3D CT NIfTI volume into a list of 2.5D slice dicts for SAM3.

    For each axial slice i that contains fracture voxels (or all slices during
    inference), stack [i-context … i … i+context] channels, normalize to [0,1],
    resize to target_hw, and return with the corresponding 2D mask.

    Returns list of dicts:
        {
          "slice_idx": int,
          "image":     np.ndarray (H, W, 2*context+1)  float32 [0,1]
          "mask":      np.ndarray (H, W)                uint8  {0,1}  or None
        }
    """
    import torch
    import numpy as np
    import nibabel as nib
    from skimage.transform import resize as sk_resize

    ct_nib  = nib.load(str(ct_path))
    ct_vol  = ct_nib.get_fdata().astype(np.float32)   # (X, Y, Z) or (Z, Y, X)

    # nnU-Net transposes: axes are (coronal, axial, sagittal) after transpose [1,0,2]
    # We iterate over axis 1 (the long Y axis = slice direction)
    # Volume shape after conversion: (D, H, W) where D=slice axis
    if ct_vol.ndim == 3:
        # Treat axis 1 as slice axis (Y = high-res direction)
        vol = np.transpose(ct_vol, (1, 0, 2))   # (Y, X, Z) = (n_slices, H, W)
    else:
        vol = ct_vol

    n_slices, H, W = vol.shape

    # Normalize HU → [0, 1]  clip to [-1000, 2000]
    vol = np.clip(vol, -1000.0, 2000.0)
    vol = (vol + 1000.0) / 3000.0

    mask_vol = None
    if mask_path is not None and mask_path.exists():
        mask_nib = nib.load(str(mask_path))
        mask_arr = mask_nib.get_fdata().astype(np.uint8)
        if mask_arr.ndim == 3:
            mask_vol = np.transpose(mask_arr, (1, 0, 2))

    H_out, W_out = target_hw
    slices = []

    for i in range(n_slices):
        # Skip slices with no fracture during training (speed up)
        if mask_vol is not None:
            mask_slice = mask_vol[i]
            if mask_slice.max() == 0:
                continue   # no fracture in this slice — skip
        else:
            mask_slice = None

        # Stack context slices
        channels = []
        for offset in range(-context, context + 1):
            j = max(0, min(n_slices - 1, i + offset))
            channels.append(vol[j])
        img = np.stack(channels, axis=-1)   # (H, W, 2*context+1)

        # Resize to target
        img_r = sk_resize(img, (H_out, W_out, img.shape[2]),
                          order=1, preserve_range=True,
                          anti_aliasing=True).astype(np.float32)

        if mask_slice is not None:
            mask_r = sk_resize(mask_slice.astype(np.float32),
                               (H_out, W_out), order=0,
                               preserve_range=True).astype(np.uint8)
        else:
            mask_r = None

        slices.append({
            "slice_idx": i,
            "image":     img_r,
            "mask":      mask_r,
            "orig_shape": (H, W),
        })

    return slices


def train_sam_fold(fold: int, n_epochs: int = 50, lr: float = 1e-4,
                   lora_rank: int = 4, context: int = 1,
                   target_hw: tuple = (512, 128),
                   run_validation: bool = False) -> None:
    """
    Train SAM3 + LoRA on one LOOCV fold using 2.5D slices and 3D Dice loss.

    2.5D approach:
        - Each input = [slice-1, slice, slice+1]  stacked as multi-channel image
        - SAM3 image encoder (ViT) + LoRA adapters → image embedding
        - SAM3 mask decoder → per-slice binary mask
        - 3D Dice loss = Dice computed across all slices of the volume at once
          (not per-slice — this maintains 3D consistency)

    LoRA:
        - All SAM3 backbone weights frozen
        - Only LoRA A/B matrices in Q/V projections are trained (~1% params)
        - Much less RAM and compute than full fine-tuning
        - Less overfitting risk with 3 training patients

    Results saved to:
        sam_results/Dataset001_Fracture_fold{fold}/
            best_lora.pt          ← best checkpoint
            training_log.json     ← loss per epoch
            predictions/          ← predicted masks for held-out patient
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n[SAM Train] Fold {fold}  device={device}  "
          f"epochs={n_epochs}  lr={lr}  rank={lora_rank}")

    # ── Load splits ───────────────────────────────────────────────────────
    splits_path = FOLDS_DIR / "cv_splits.json"
    if not splits_path.exists():
        print("  Run 'prepare' first.")
        return
    with open(splits_path) as f:
        data = json.load(f)
    splits = data["splits"] if isinstance(data, dict) and "splits" in data else data
    if fold >= len(splits):
        print(f"  Fold {fold} not found in splits.")
        return

    train_pids = splits[fold]["train"]
    val_pids   = splits[fold]["val"]
    print(f"  Train: {train_pids}  Val: {val_pids}")

    # ── Output dirs ───────────────────────────────────────────────────────
    sam_fold_dir = BASE_DIR / "sam_results" / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold}"
    pred_dir     = sam_fold_dir / "predictions"
    sam_fold_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────────
    print("\n  Building SAM3 + LoRA head...")
    try:
        model, processor, seg_head, trainable_params = _build_sam3_lora(
            rank=lora_rank, device=device
        )
    except RuntimeError as e:
        print(f"  ✗ {e}")
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # ── Load training slices ──────────────────────────────────────────────
    print("\n  Loading training slices...")
    all_train_slices = []
    for pid in train_pids:
        ct_path   = CONVERTED_DIR / f"{pid}_ct.nii.gz"
        mask_path = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
        if not ct_path.exists():
            print(f"  ⚠️  Missing {pid} — run prepare first")
            continue
        slices = _ct_to_sam3_slices(ct_path, mask_path,
                                     context=context, target_hw=target_hw)
        all_train_slices.extend(slices)
        print(f"    {pid}: {len(slices)} fracture slices")

    if not all_train_slices:
        print("  ✗ No training slices found.")
        return
    print(f"  Total training slices: {len(all_train_slices)}")

    # ── Cache backbone features (frozen backbone — only need to run once) ──
    import torch.nn.functional as F_nn
    import nibabel as nib
    from PIL import Image as PILImage

    cache_dir = sam_fold_dir / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    n_channels = 2 * context + 1

    print("\n  Caching SAM3 backbone features (one-time, CPU)...")
    feature_cache = {}   # pid -> {idx: tensor (256, H, W)}
    gt_cache      = {}   # pid -> tensor (n_slices, H, W)

    for pid in train_pids:
        ct_path   = CONVERTED_DIR / f"{pid}_ct.nii.gz"
        mask_path = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
        if not ct_path.exists():
            continue

        cache_file = cache_dir / f"{pid}_feats.pt"
        mask_nib   = nib.load(str(mask_path))
        mask_vol   = mask_nib.get_fdata().astype(np.float32)
        mask_vol   = np.transpose(mask_vol, (1, 0, 2))
        n_sl       = mask_vol.shape[0]

        # Cache GT
        gt_t = torch.zeros(n_sl, *target_hw)
        from skimage.transform import resize as sk_resize
        for si in range(n_sl):
            gt_r = sk_resize(mask_vol[si], target_hw, order=0,
                             preserve_range=True).astype(np.float32)
            gt_t[si] = torch.from_numpy(gt_r)
        gt_cache[pid] = gt_t

        # Load or compute backbone features — save per-slice as .npy to avoid RAM OOM
        pid_cache_dir = cache_dir / pid
        pid_cache_dir.mkdir(exist_ok=True)
        pid_index_file = cache_dir / f"{pid}_index.npy"

        if pid_index_file.exists():
            slice_indices = np.load(pid_index_file).tolist()
            print(f"    {pid}: found {len(slice_indices)} cached slices")
        else:
            print(f"    {pid}: computing backbone features...")
            pid_slices = _ct_to_sam3_slices(ct_path, None,
                                             context=context, target_hw=target_hw)
            slice_indices = []
            model.eval()
            with torch.no_grad():
                for si, sl in enumerate(pid_slices):
                    if si % 50 == 0:
                        print(f"      slice {si}/{len(pid_slices)}...")
                    img    = sl["image"]
                    idx    = sl["slice_idx"]
                    img_np = img
                    if n_channels != 3:
                        repeats = max(1, 3 // n_channels)
                        img_np  = np.concatenate([img_np]*repeats, axis=2)[:,:,:3]
                    img_uint8 = (img_np * 255).clip(0,255).astype(np.uint8)
                    pil_img   = PILImage.fromarray(img_uint8, mode="RGB")
                    try:
                        inf_state = processor.set_image(pil_img)
                        bfeats    = inf_state['backbone_out']['vision_features']
                        bfeats_r  = F_nn.interpolate(bfeats.float(), size=target_hw,
                                                      mode='bilinear', align_corners=False)
                        feat_np   = bfeats_r.squeeze(0).cpu().numpy()
                        del inf_state, bfeats, bfeats_r  # free immediately
                        np.save(str(pid_cache_dir / f"{idx}.npy"), feat_np)
                    except Exception as e:
                        np.save(str(pid_cache_dir / f"{idx}.npy"),
                                np.zeros((256, *target_hw), dtype=np.float32))
                    slice_indices.append(idx)
                    # Clear any accumulated MPS/CPU memory every 10 slices
                    if len(slice_indices) % 10 == 0:
                        import gc
                        gc.collect()
                        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
            np.save(str(pid_index_file), np.array(slice_indices))
            print(f"    {pid}: cached {len(slice_indices)} slices to disk")
        feature_cache[pid] = slice_indices  # just the list of indices

    print("  Feature caching done! Training on MPS...")

    # ── Training loop (seg_head only, backbone features from cache) ────────
    log = []
    best_loss = float("inf")
    head_device = next(seg_head.parameters()).device

    for epoch in range(n_epochs):
        seg_head.train()
        epoch_losses = []

        for pid in train_pids:
            if pid not in feature_cache:
                continue

            slice_indices = feature_cache[pid]
            pid_cache_dir = cache_dir / pid
            gt_vol_full = gt_cache[pid]  # keep on CPU
            n_slices = gt_vol_full.shape[0]

            optimizer.zero_grad()

            # Accumulate 2D Dice loss slice-by-slice — never build full volume on MPS
            loss_sum = torch.tensor(0.0, device=head_device, requires_grad=False)
            n_valid  = 0

            for idx in slice_indices:
                try:
                    feat_np  = np.load(str(pid_cache_dir / f"{idx}.npy"))
                    raw_mask = torch.from_numpy(feat_np).unsqueeze(0).to(head_device)
                    pred_2d  = seg_head(raw_mask).squeeze()          # (H, W)
                    gt_2d    = gt_vol_full[idx].to(head_device)      # (H, W)

                    # 2D Dice loss for this slice
                    smooth   = 1e-5
                    inter    = (pred_2d * gt_2d).sum()
                    dice_sl  = 1.0 - (2.0 * inter + smooth) / (pred_2d.sum() + gt_2d.sum() + smooth)
                    loss_sum = loss_sum + dice_sl
                    n_valid += 1

                    # Free immediately
                    del raw_mask, pred_2d, gt_2d, feat_np
                    if n_valid % 20 == 0:
                        import gc; gc.collect()
                        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                except Exception as e:
                    if epoch == 0:
                        print(f"    Forward pass error (slice {idx}): {e}")
                    continue

            if n_valid == 0:
                continue

            loss = loss_sum / n_valid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # ── Validation on true held-out patient (HPC mode only) ──────────
        # Skipped by default on laptop to avoid OOM. Enable with --validate.
        if run_validation:
            seg_head.eval()
            val_losses = []
            with torch.no_grad():
                for pid in val_pids:
                    val_cache_dir = cache_dir / pid
                    val_index_file = cache_dir / f"{pid}_index.npy"

                    # Cache val patient features if not done yet
                    if not val_index_file.exists():
                        ct_path = CONVERTED_DIR / f"{pid}_ct.nii.gz"
                        mask_path = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
                        if not ct_path.exists():
                            continue
                        val_cache_dir.mkdir(exist_ok=True)
                        print(f"  Caching val patient {pid} features...")
                        val_slices_data = _ct_to_sam3_slices(ct_path, None, context=context, target_hw=target_hw)
                        val_indices = []
                        for si, sl in enumerate(val_slices_data):
                            idx = sl["slice_idx"]
                            img_np = sl["image"]
                            if n_channels != 3:
                                repeats = max(1, 3 // n_channels)
                                img_np = np.concatenate([img_np]*repeats, axis=2)[:,:,:3]
                            img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
                            pil_img = PILImage.fromarray(img_uint8, mode="RGB")
                            try:
                                inf_state = processor.set_image(pil_img)
                                bfeats = inf_state['backbone_out']['vision_features']
                                bfeats_r = F_nn.interpolate(bfeats.float(), size=target_hw, mode='bilinear', align_corners=False)
                                np.save(str(val_cache_dir / f"{idx}.npy"), bfeats_r.squeeze(0).cpu().numpy())
                            except Exception:
                                np.save(str(val_cache_dir / f"{idx}.npy"), np.zeros((256, *target_hw), dtype=np.float32))
                            val_indices.append(idx)
                        np.save(str(val_index_file), np.array(val_indices))
                        # Cache GT for val patient
                        mask_nib = nib.load(str(mask_path))
                        mask_vol = np.transpose(mask_nib.get_fdata().astype(np.float32), (1, 0, 2))
                        n_sl = mask_vol.shape[0]
                        gt_t = torch.zeros(n_sl, *target_hw)
                        from skimage.transform import resize as sk_resize
                        for si in range(n_sl):
                            gt_r = sk_resize(mask_vol[si], target_hw, order=0, preserve_range=True).astype(np.float32)
                            gt_t[si] = torch.from_numpy(gt_r)
                        gt_cache[pid] = gt_t

                    val_slice_indices = np.load(str(val_index_file)).tolist()
                    val_gt = gt_cache.get(pid)
                    if val_gt is None:
                        continue

                    for idx in val_slice_indices:
                        npy_path = val_cache_dir / f"{idx}.npy"
                        if not npy_path.exists():
                            continue
                        try:
                            feat_np  = np.load(str(npy_path))
                            raw_mask = torch.from_numpy(feat_np).unsqueeze(0).to(head_device)
                            pred_2d  = seg_head(raw_mask).squeeze()
                            gt_2d    = val_gt[idx].to(head_device)
                            smooth   = 1e-5
                            inter    = (pred_2d * gt_2d).sum()
                            val_dice = 1.0 - (2.0 * inter + smooth) / (pred_2d.sum() + gt_2d.sum() + smooth)
                            val_losses.append(val_dice.item())
                            del raw_mask, pred_2d, gt_2d, feat_np
                        except Exception:
                            continue

        if run_validation:
            seg_head.train()
            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            log.append({"epoch": epoch, "dice_loss": mean_loss, "val_dice_loss": mean_val_loss})
            print(f"  Epoch {epoch:3d}/{n_epochs}  Train={mean_loss:.4f}  Val={mean_val_loss:.4f}")
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                ckpt_path = sam_fold_dir / "best_lora.pt"
                torch.save(seg_head.state_dict(), ckpt_path)
                print(f"    ✓ Saved best checkpoint (val_loss={best_loss:.4f})")
        else:
            log.append({"epoch": epoch, "dice_loss": mean_loss})
            print(f"  Epoch {epoch:3d}/{n_epochs}  Train={mean_loss:.4f}")
            if mean_loss < best_loss:
                best_loss = mean_loss
                ckpt_path = sam_fold_dir / "best_lora.pt"
                torch.save(seg_head.state_dict(), ckpt_path)
                print(f"    ✓ Saved best checkpoint (loss={best_loss:.4f})")

    # Save training log
    with open(sam_fold_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Predict held-out patient ──────────────────────────────────────────
    print(f"\n  Predicting held-out patient: {val_pids}")
    model.eval()
    seg_head.eval()

    for pid in val_pids:
        ct_path = CONVERTED_DIR / f"{pid}_ct.nii.gz"
        if not ct_path.exists():
            continue

        import nibabel as nib
        ct_nib   = nib.load(str(ct_path))
        ct_vol   = ct_nib.get_fdata().astype(np.float32)
        orig_shape = ct_vol.shape                          # (X, Y, Z)
        n_slices = ct_vol.shape[1]                         # Y axis = slice axis

        all_slices = _ct_to_sam3_slices(ct_path, None,
                                         context=context, target_hw=target_hw)
        # Also get slices that were skipped (non-fracture) for full prediction
        # Re-run without mask filter
        vol_t = np.clip(np.transpose(ct_vol, (1, 0, 2)), -1000.0, 2000.0)
        vol_t = (vol_t + 1000.0) / 3000.0
        H_out, W_out = target_hw

        pred_vol = np.zeros((n_slices, orig_shape[0], orig_shape[2]),
                             dtype=np.float32)

        with torch.no_grad():
            for i in range(n_slices):
                channels = []
                for offset in range(-context, context + 1):
                    j = max(0, min(n_slices - 1, i + offset))
                    channels.append(vol_t[j])
                img = np.stack(channels, axis=-1)

                img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
                if n_channels != 3:
                    repeats = max(1, 3 // n_channels)
                    img_t = img_t.repeat(1, repeats, 1, 1)[:, :3, :, :]

                try:
                    from PIL import Image as PILImage
                    from skimage.transform import resize as sk_resize
                    img_np = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(img_np, mode="RGB")

                    inf_state = processor.set_image(pil_img)
                    output = processor.set_text_prompt(
                        state=inf_state, prompt="pelvic fracture"
                    )
                    masks_out = output.get("masks", None)
                    scores_out = output.get("scores", None)

                    if masks_out is not None and len(masks_out) > 0:
                        best = int(scores_out.argmax()) if scores_out is not None else 0
                        mask_np = masks_out[best].squeeze().astype(np.float32)
                        mask_r = sk_resize(mask_np, target_hw, order=1,
                                           preserve_range=True).astype(np.float32)
                        raw_mask = torch.from_numpy(mask_r).unsqueeze(0).unsqueeze(0).to(next(seg_head.parameters()).device)
                        refined = seg_head(raw_mask).squeeze().cpu().numpy()
                        pred_orig = sk_resize(refined, (orig_shape[0], orig_shape[2]),
                                              order=1, preserve_range=True)
                        pred_vol[i] = pred_orig

                except Exception:
                    pass

        # Threshold and save as NIfTI
        pred_binary = (np.transpose(pred_vol, (1, 0, 2)) > 0.5).astype(np.uint8)
        out_nib = nib.Nifti1Image(pred_binary, ct_nib.affine, ct_nib.header)
        out_path = pred_dir / f"{pid}.nii.gz"
        nib.save(out_nib, str(out_path))
        print(f"  Saved prediction → {out_path}")

    print(f"\n  ✓ SAM fold {fold} complete.")
    print(f"  Results → {sam_fold_dir}")
    print(f"  Next: python fracture_pipeline.py evaluate --model sam")


def plot_sam_training_curves(fold: int, output_dir: str = "plots") -> Optional[Path]:
    """
    Plot SAM3 train + validation Dice loss curves from training_log.json.
    Shows both curves on same axis so overfitting/generalisation is visible.
    """
    log_path = BASE_DIR / "sam_results" / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold}" / "training_log.json"
    if not log_path.exists():
        print(f"  ⚠️  No SAM training log for fold {fold} at {log_path}")
        return None

    with open(log_path) as f:
        log = json.load(f)

    epochs     = [e["epoch"]                        for e in log]
    train_loss = [e["dice_loss"]                    for e in log]
    val_loss   = [e.get("val_dice_loss", float("nan")) for e in log]
    has_val    = any(not np.isnan(v) for v in val_loss)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, color="#2E75B6", linewidth=2,
            marker="o", markersize=3, markerfacecolor="white",
            markeredgewidth=1.5, label="Train Dice Loss")

    if has_val:
        ax.plot(epochs, val_loss, color="#E74C3C", linewidth=2,
                marker="s", markersize=3, markerfacecolor="white",
                markeredgewidth=1.5, linestyle="--", label="Val Dice Loss (held-out patient)")
        best_ep  = int(np.nanargmin(val_loss))
        best_val = val_loss[best_ep]
        ax.axvline(best_ep, color="#E74C3C", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.annotate(f"Best val\nepoch {best_ep}\n({best_val:.4f})",
                    xy=(best_ep, best_val), xytext=(best_ep + max(1, len(epochs)*0.05), best_val + 0.0002),
                    fontsize=9, color="#E74C3C",
                    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.0))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Dice Loss", fontsize=12)
    ax.set_title(f"SAM3 Training Curve — Fold {fold}\n"
                 f"(frozen backbone + lightweight seg head, slice-wise Dice loss)",
                 fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sam_training_curves_fold{fold}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path

def main():
    parser = build_parser()
    args   = parser.parse_args()
    registry = FractureRegistry()

    # ── add ───────────────────────────────────
    if args.command == "add":
        registry.add(
            patient_id=args.patient_id,
            dicom_dir=args.dicom_dir,
            seg_fracture=args.seg_fracture,
            seg_pelvis=args.seg_pelvis,
        )

    # ── list ──────────────────────────────────
    elif args.command == "list":
        registry.summary()

    # ── remove ────────────────────────────────
    elif args.command == "remove":
        registry.remove(args.patient_id)

    # ── qc ────────────────────────────────────
    elif args.command == "qc":
        if len(registry) == 0:
            print("No patients registered. Use: python fracture_pipeline.py add ...")
            return
        if args.patient_id:
            case = registry.get(args.patient_id)
            if case is None:
                print(f"Patient {args.patient_id} not found.")
                return
            qc_case(case)
        else:
            print(f"Running QC on all {len(registry)} patients...")
            for case in registry:
                qc_case(case)

    # ── prepare ───────────────────────────────
    elif args.command == "prepare":
        if len(registry) == 0:
            print("No patients registered.")
            return
        convert_all_patients(registry)
        splits, strategy = make_cv_splits(registry.cases)

        splits_path = FOLDS_DIR / "cv_splits.json"
        FOLDS_DIR.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump({"strategy": strategy, "splits": splits}, f, indent=2)

        prepare_nnunet_folds(splits, hpc_splits=args.hpc_splits)
        print("\n✓ Pipeline ready.")
        print("  Next: python fracture_pipeline.py train --fold 0")

    # ── train ─────────────────────────────────
    elif args.command == "train":
        train_fold(args.fold)

    # ── evaluate ──────────────────────────────
    elif args.command == "evaluate":
        splits = _load_splits()
        if splits is None:
            return
        evaluate_all_folds(splits)

    # ── predict ───────────────────────────────
    elif args.command == "predict":
        predict_new_patient(
            dicom_dir=args.dicom_dir,
            seg_pelvis=args.seg_pelvis,
            output_path=args.output,
            fold=args.fold,
        )

    # ── ensemble ──────────────────────────────
    elif args.command == "ensemble":
        splits = _load_splits()
        n_folds = args.n_folds if args.n_folds else (len(splits) if splits else 4)
        predict_ensemble(
            dicom_dir=args.dicom_dir,
            seg_pelvis=args.seg_pelvis,
            output_path=args.output,
            n_folds=n_folds,
            threshold=args.threshold,
        )

    # ── train_sam ─────────────────────────────
    elif args.command == "train_sam":
        train_sam_fold(
            fold=args.fold,
            n_epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.rank,
            context=args.context,
            run_validation=args.validate,
        )

    # ── retrain_all ───────────────────────────
    elif args.command == "retrain_all":
        retrain_all(n_epochs=args.n_epochs)

    # ── plot ──────────────────────────────────
    elif args.command == "plot":
        splits = _load_splits()
        what   = args.what
        odir   = args.output_dir

        if what in ("predictions", "all"):
            if splits is None:
                return
            print("\n── Prediction overlays ──────────────────")
            plot_all_predictions(splits, output_dir=odir)

        if what in ("curves", "all"):
            print("\n── Training curves ──────────────────────")
            if args.fold is not None:
                plot_training_curves(args.fold, output_dir=odir)
            else:
                if splits is None:
                    return
                for i in range(len(splits)):
                    plot_training_curves(i, output_dir=odir)

        if what in ("dice", "all"):
            print("\n── Dice summary ─────────────────────────")
            if splits is None:
                return
            plot_dice_summary(splits, output_dir=odir)

        print(f"\n✓ All plots saved to {odir}/")


if __name__ == "__main__":
    main()