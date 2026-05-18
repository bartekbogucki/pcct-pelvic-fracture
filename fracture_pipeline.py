"""
fracture_pipeline.py
====================
Two-stage CT pelvic fracture segmentation pipeline.

Stage 1 (skipped — you have pelvis annotations):
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
    Add patients as annotations arrive — pipeline stays unchanged.
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
    Load CT DICOM folder → (volume_HU, affine_4x4, spacing_ZYX).
    volume_HU shape: (Z, H, W)  float32
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
    Load segmentation DICOM → binary numpy (Z, H, W) uint8.
    Handles both standard DICOM SEG and simple multi-frame DICOMs.
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
    Crop volume (and optionally a second mask) to bounding box of
    `mask` + `padding` voxels.

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
    NIfTI convention is (X, Y, Z) — we transpose accordingly.
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
        try:
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

            # Sanity checks
            if pelvis.sum() == 0:
                print(f"  ⚠️  Pelvis mask is EMPTY for {case.patient_id} — skipping.")
                continue
            if fracture.sum() == 0:
                print(f"  ⚠️  Fracture mask is EMPTY for {case.patient_id} — skipping.")
                continue

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
        except Exception as e:
            print(f"  ✗ Failed {case.patient_id}: {type(e).__name__}: {e}")
            print(f"    Skipping and continuing...")
            continue

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
        # With hpc_splits: held-out patient is also in imagesTr for val loss
        # so numTraining must reflect the actual number of files in imagesTr
        n_training = len(split["train"]) + (len(split["val"]) if hpc_splits else 0)
        dataset_json = {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "fracture": 1},
            "numTraining": n_training,
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


def precision_recall(pred: np.ndarray, gt: np.ndarray,
                     smooth: float = 1e-6) -> Tuple[float, float]:
    """
    Precision = TP / (TP + FP)  — how many predicted voxels are correct
    Recall    = TP / (TP + FN)  — how many GT voxels are found
    """
    pred = (pred > 0).astype(np.float32).flatten()
    gt   = (gt   > 0).astype(np.float32).flatten()
    tp   = (pred * gt).sum()
    fp   = (pred * (1 - gt)).sum()
    fn   = ((1 - pred) * gt).sum()
    precision = float((tp + smooth) / (tp + fp + smooth))
    recall    = float((tp + smooth) / (tp + fn + smooth))
    return precision, recall


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                           spacing: Tuple = (1.0, 1.0, 1.0)) -> float:
    """
    95th percentile Hausdorff Distance in mm.
    Measures boundary accuracy — lower is better.
    Returns inf if either mask is empty.
    Crops to bounding box of union for speed on large volumes.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion
    pred_b = (pred > 0).astype(bool)
    gt_b   = (gt   > 0).astype(bool)

    if not pred_b.any() or not gt_b.any():
        return float("inf")

    # Crop to bounding box of union for speed
    union  = pred_b | gt_b
    coords = np.argwhere(union)
    pad    = 5
    z0, y0, x0 = np.maximum(coords.min(axis=0) - pad, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + pad + 1,
                             np.array(pred_b.shape))
    pred_b = pred_b[z0:z1, y0:y1, x0:x1]
    gt_b   = gt_b[z0:z1,   y0:y1, x0:x1]

    # Distance transforms on cropped region only
    pred_dt = distance_transform_edt(~pred_b, sampling=spacing)
    gt_dt   = distance_transform_edt(~gt_b,   sampling=spacing)

    # Border voxels
    pred_border = pred_b & ~binary_erosion(pred_b)
    gt_border   = gt_b   & ~binary_erosion(gt_b)

    dist_pred_to_gt = gt_dt[pred_border]
    dist_gt_to_pred = pred_dt[gt_border]

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def surface_dice(pred: np.ndarray, gt: np.ndarray,
                 spacing: Tuple = (1.0, 1.0, 1.0),
                 tolerance: float = 1.0) -> float:
    """
    Surface Dice at tolerance (mm).
    Measures overlap of surfaces within a tolerance distance.
    tolerance=1.0mm means surfaces within 1mm count as matching.
    Returns value in [0, 1] — higher is better.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion
    pred_b = (pred > 0).astype(bool)
    gt_b   = (gt   > 0).astype(bool)

    if not pred_b.any() or not gt_b.any():
        return 0.0

    # Surface voxels
    pred_border = pred_b & ~binary_erosion(pred_b)
    gt_border   = gt_b   & ~binary_erosion(gt_b)

    # Distance transforms
    pred_dt = distance_transform_edt(~pred_b, sampling=spacing)
    gt_dt   = distance_transform_edt(~gt_b,   sampling=spacing)

    # Surface voxels within tolerance of the other surface
    pred_border_within = pred_dt[gt_border]  <= tolerance
    gt_border_within   = gt_dt[pred_border]  <= tolerance

    numerator   = pred_border_within.sum() + gt_border_within.sum()
    denominator = gt_border.sum() + pred_border.sum()

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def evaluate_all_folds(splits: List[Dict], model: str = "nnunet",
                       eval_set: str = "val") -> None:
    """
    Load predictions and compute comprehensive metrics per fold:
        - 3D Dice
        - Precision & Recall
        - HD95 (Hausdorff Distance 95th percentile)
        - Surface Dice (tolerance=1mm)

    Parameters
    ----------
    model:    "nnunet" or "sam"
    eval_set: "val"   — evaluate on per-fold validation patients (for CV metrics)
              "test"  — evaluate on held-out test patients (for final metrics)
                        ⚠️  Only the best fold's predictions are typically reported
    """
    if eval_set not in ("val", "test"):
        raise ValueError(f"eval_set must be 'val' or 'test', got {eval_set}")

    print(f"\n[Evaluate] {len(splits)} folds  model={model}  eval_set={eval_set}")
    all_dice, all_hd95, all_prec, all_rec, all_sdice = [], [], [], [], []
    results = []

    for fold_i, split in enumerate(splits):
        if model == "nnunet":
            pred_dir = (RESULTS_DIR /
                        f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold_i}" /
                        "predictions_3d_fullres")
        else:
            # SAM — find most recent experiment folder for this fold
            sam_base = BASE_DIR / "sam_results"
            candidates = sorted(sam_base.glob(
                f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold_i}*/predictions"
            ))
            if not candidates:
                print(f"  ⚠️  No SAM results for fold {fold_i}")
                continue
            pred_dir = candidates[-1]
            print(f"  SAM results: {pred_dir.parent.name}")

        # Choose which patients to evaluate
        eval_pids = split.get(eval_set, [])
        if not eval_pids:
            if eval_set == "test":
                print(f"  ⚠️  No test patients in splits (need 20+ patients)")
            continue

        for pid in eval_pids:
            pred_path = pred_dir / f"{pid}.nii.gz"
            gt_path   = CONVERTED_DIR / f"{pid}_fracture.nii.gz"

            if not pred_path.exists():
                print(f"  ⚠️  No prediction for {pid} at {pred_path}")
                continue
            if not gt_path.exists():
                print(f"  ⚠️  No ground truth for {pid}")
                continue

            pred_nib = nib.load(pred_path)
            pred     = pred_nib.get_fdata()
            gt       = nib.load(gt_path).get_fdata()

            # Get voxel spacing from NIfTI header (mm)
            zooms   = pred_nib.header.get_zooms()[:3]
            spacing = tuple(float(z) for z in zooms)

            # Compute all metrics
            d              = dice_3d(pred, gt)
            prec, rec      = precision_recall(pred, gt)
            hd95           = hausdorff_distance_95(pred, gt, spacing=spacing)
            sdice          = surface_dice(pred, gt, spacing=spacing, tolerance=1.0)

            all_dice.append(d)
            all_prec.append(prec)
            all_rec.append(rec)
            all_sdice.append(sdice)
            if not np.isinf(hd95):
                all_hd95.append(hd95)

            print(f"  Fold {fold_i}  {pid}")
            print(f"    Dice:          {d:.4f}")
            print(f"    Precision:     {prec:.4f}")
            print(f"    Recall:        {rec:.4f}")
            print(f"    HD95 (mm):     {hd95:.2f}" if not np.isinf(hd95) else "    HD95 (mm):     ∞ (empty pred)")
            print(f"    Surface Dice:  {sdice:.4f}")

            results.append({
                "fold":         fold_i,
                "patient_id":   pid,
                "dice_3d":      round(d, 4),
                "precision":    round(prec, 4),
                "recall":       round(rec, 4),
                "hd95_mm":      round(hd95, 2) if not np.isinf(hd95) else None,
                "surface_dice": round(sdice, 4),
            })

    if all_dice:
        print(f"\n  {'─'*50}")
        print(f"  {'Metric':<20} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'─'*50}")
        print(f"  {'Dice':<20} {np.mean(all_dice):>8.4f}  {np.std(all_dice):>8.4f}  {np.min(all_dice):>8.4f}  {np.max(all_dice):>8.4f}")
        print(f"  {'Precision':<20} {np.mean(all_prec):>8.4f}  {np.std(all_prec):>8.4f}  {np.min(all_prec):>8.4f}  {np.max(all_prec):>8.4f}")
        print(f"  {'Recall':<20} {np.mean(all_rec):>8.4f}  {np.std(all_rec):>8.4f}  {np.min(all_rec):>8.4f}  {np.max(all_rec):>8.4f}")
        print(f"  {'Surface Dice':<20} {np.mean(all_sdice):>8.4f}  {np.std(all_sdice):>8.4f}  {np.min(all_sdice):>8.4f}  {np.max(all_sdice):>8.4f}")
        if all_hd95:
            print(f"  {'HD95 (mm)':<20} {np.mean(all_hd95):>8.2f}  {np.std(all_hd95):>8.2f}  {np.min(all_hd95):>8.2f}  {np.max(all_hd95):>8.2f}")
        print(f"  {'─'*50}")
        print(f"  N patients: {len(all_dice)}")

        out_path = RESULTS_DIR / f"evaluation_results_{model}.json"
        with open(out_path, "w") as f:
            json.dump({
                "model":       model,
                "per_patient": results,
                "summary": {
                    "mean_dice":         round(float(np.mean(all_dice)),   4),
                    "std_dice":          round(float(np.std(all_dice)),    4),
                    "mean_precision":    round(float(np.mean(all_prec)),   4),
                    "mean_recall":       round(float(np.mean(all_rec)),    4),
                    "mean_surface_dice": round(float(np.mean(all_sdice)),  4),
                    "mean_hd95_mm":      round(float(np.mean(all_hd95)),   2) if all_hd95 else None,
                    "n":                 len(all_dice),
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
    ev = sub.add_parser("evaluate", help="Compute comprehensive metrics for all folds")
    ev.add_argument("--model", choices=["nnunet", "sam"], default="nnunet",
                    help="Which model predictions to evaluate (default: nnunet)")
    ev.add_argument("--eval_set", choices=["val", "test"], default="val",
                    help="Evaluate on validation (per-fold) or held-out test set. "
                         "Use 'test' for final evaluation after training (default: val)")

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
    ts.add_argument("--loss", default="dice_3d_ce_boundary",
                    choices=["dice", "dice_ce", "dice_ce_boundary",
                             "dice_3d", "dice_3d_ce", "dice_3d_ce_boundary"],
                    help=(
                        "Training loss function:\n"
                        "  dice                  — plain per-slice Dice (legacy)\n"
                        "  dice_ce               — per-slice Dice + CE\n"
                        "  dice_ce_boundary      — per-slice Dice + CE + 2D Boundary\n"
                        "  dice_3d               — true 3D Dice over full volume\n"
                        "  dice_3d_ce            — 3D Dice + 3D CE  [HPC v2 ablation]\n"
                        "  dice_3d_ce_boundary   — 3D Dice + 3D CE + 3D Boundary  [HPC v1, laptop]\n"
                        "Default: dice_3d_ce_boundary"
                    ))
    ts.add_argument("--input_mode", default="2.5d",
                    choices=["2.5d", "single"],
                    help=(
                        "Slice input mode:\n"
                        "  2.5d   — stack slice ± context neighbours as channels  [default]\n"
                        "  single — each slice processed independently  [HPC v3 ablation]"
                    ))
    ts.add_argument("--postprocess", action="store_true", default=False,
                    help="Apply 3D post-processing after inference "
                         "(connected component cleaning + gap bridging). "
                         "Recommended for HPC.")
    ts.add_argument("--sampling", default="weighted_ce",
                    choices=["full_slice", "weighted_ce", "focal", "patch"],
                    help=(
                        "Strategy to handle extreme class imbalance (<0.01%% fracture voxels).\n"
                        "\n"
                        "  weighted_ce  — DEFAULT. Full slice + weighted CE. Fracture voxels\n"
                        "                 upweighted by inverse frequency (clipped at --ce_weight).\n"
                        "                 Keeps 3D Dice + boundary loss fully intact.\n"
                        "                 Best combined with --loss dice_3d_ce_boundary.\n"
                        "\n"
                        "  full_slice   — Full slice, plain unweighted CE. Baseline only.\n"
                        "                 CE dominated by background (99.99%% of voxels).\n"
                        "\n"
                        "  focal        — Full slice, focal loss replaces CE. Automatically\n"
                        "                 down-weights easy background voxels. No manual tuning.\n"
                        "                 Keeps 3D Dice + boundary loss.\n"
                        "\n"
                        "  patch        — nnU-Net-style 33%% fg patch oversampling. Strongest\n"
                        "                 class balance but incompatible with 3D Dice and\n"
                        "                 boundary loss (patches cannot be assembled into volume).\n"
                        "                 Use --loss dice for patch mode."
                    ))
    ts.add_argument("--ce_weight", type=float, default=50.0,
                    help="Foreground weight for weighted CE (--sampling weighted_ce). "
                         "Default 50 — each fracture voxel = 50x background voxels. "
                         "Use 'auto' via --ce_weight 0 to compute from inverse frequency.")
    ts.add_argument("--patch_size", type=int, default=64,
                    help="Patch size for patch oversampling (--sampling patch). Default 64.")
    ts.add_argument("--n_patches", type=int, default=8,
                    help="Number of patches per slice for patch oversampling. Default 8.")
    ts.add_argument("--patch_ce_weight", type=float, default=0.0,
                    help="Apply weighted CE to patch sampling too (0=disabled). "
                         "Combines patch oversampling + weighted CE like nnU-Net. "
                         "Example: --sampling patch --patch_ce_weight 50")
    ts.add_argument("--fg_fraction", type=float, default=0.33,
                    help=(
                        "Fraction of patches centred on fracture voxels (--sampling patch).\n"
                        "  0.33 — nnU-Net standard: 33%% fg, 67%% random  (default)\n"
                        "  0.50 — aggressive: equal fg/bg patches\n"
                        "  0.00 — pure random (no oversampling, baseline)"
                    ))
    ts.add_argument("--use_text_prompt", action="store_true", default=False,
                    help="Use 'pelvic fracture' text prompt → SAM3 mask decoder output. "
                         "When False: backbone features → conv head (default).")
    ts.add_argument("--use_medsam3", action="store_true", default=False,
                    help="Load MedSAM3 medical LoRA weights before fracture fine-tuning. "
                         "Gives better starting point than raw SAM3. "
                         "Weights auto-downloaded from huggingface.co/lal-Joey/MedSAM3_v1")
    ts.add_argument("--exp_name", type=str, default=None,
                    help=(
                        "Experiment name — appended to output folder so different runs "
                        "don\'t overwrite each other.\n"
                        "Default: auto-generated from key hyperparameters.\n"
                        "Example: --exp_name weighted_ce_rank4_ctx1"
                    ))

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


def _boundary_loss_2d(pred: "torch.Tensor", gt_np: "np.ndarray",
                      spacing: tuple = (1.0, 1.0)) -> "torch.Tensor":
    """
    Boundary loss for a single 2D slice (Kervadec et al. 2019, MedIA 2021).
    Used in laptop mode where building full 3D volume is not feasible.

    pred    : (H, W)  float tensor, sigmoid probabilities
    gt_np   : (H, W)  numpy uint8, ground truth binary mask
    spacing : (dy, dx) voxel spacing in mm

    Reference:
        Kervadec et al. (2021). Boundary loss for highly unbalanced segmentation.
        Medical Image Analysis, 67, 101851.
    """
    import torch
    from scipy.ndimage import distance_transform_edt

    gt_bool = gt_np.astype(bool)
    if not gt_bool.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=False)

    dist_outside = distance_transform_edt(~gt_bool, sampling=spacing)
    dist_inside  = distance_transform_edt( gt_bool, sampling=spacing)
    phi_G = dist_outside - dist_inside
    max_dist = np.abs(phi_G).max()
    if max_dist > 0:
        phi_G = phi_G / max_dist
    phi_tensor = torch.from_numpy(phi_G.astype(np.float32)).to(pred.device)
    return (phi_tensor * pred).mean()


def _boundary_loss_3d(pred_vol: "torch.Tensor", gt_vol_np: "np.ndarray",
                      spacing: tuple = (1.0, 1.0, 1.0)) -> "torch.Tensor":
    """
    True 3D boundary loss (Kervadec et al. 2021) over the full assembled volume.

    Computes the signed distance map in 3D space with physical voxel spacing,
    so inter-slice distances are correctly weighted.  For thin fracture surfaces
    (mean 0.57mm annotation thickness) this is more accurate than per-slice 2D
    boundary loss because the fracture crack extends continuously across slices.

    The gradient phi_G(v) always points toward the GT surface — never vanishes —
    which addresses the vanishing-gradient problem of Dice loss on sparse labels.

    pred_vol  : (N, H, W)  float tensor on device
    gt_vol_np : (N, H, W)  numpy uint8
    spacing   : (dz, dy, dx) physical voxel spacing in mm
                dz = slice spacing (Y axis, 0.100mm for your PCCT)
                dy = in-plane (X axis, 0.577mm)
                dx = in-plane (Z axis, 0.577mm)

    Reference:
        Kervadec et al. (2021). Boundary loss for highly unbalanced segmentation.
        Medical Image Analysis, 67, 101851. doi:10.1016/j.media.2020.101851
    """
    import torch
    from scipy.ndimage import distance_transform_edt

    gt_bool = gt_vol_np.astype(bool)
    if not gt_bool.any():
        return torch.tensor(0.0, device=pred_vol.device, requires_grad=False)

    # 3D signed distance map — computed once per volume, not per slice
    dist_outside = distance_transform_edt(~gt_bool, sampling=spacing)
    dist_inside  = distance_transform_edt( gt_bool, sampling=spacing)
    phi_G = dist_outside - dist_inside    # (N, H, W) float64

    # Normalise to [-1, 1]
    max_dist = np.abs(phi_G).max()
    if max_dist > 0:
        phi_G = phi_G / max_dist

    phi_tensor = torch.from_numpy(phi_G.astype(np.float32)).to(pred_vol.device)
    return (phi_tensor * pred_vol).mean()


def _ce_loss_2d(pred: "torch.Tensor", gt: "torch.Tensor",
                eps: float = 1e-7) -> "torch.Tensor":
    """Binary cross-entropy for one 2D slice."""
    import torch
    pred = pred.clamp(eps, 1.0 - eps)
    return -(gt * torch.log(pred) + (1.0 - gt) * torch.log(1.0 - pred)).mean()


def _ce_loss_3d(pred_vol: "torch.Tensor", gt_vol: "torch.Tensor",
                eps: float = 1e-7) -> "torch.Tensor":
    """Binary cross-entropy over full 3D volume."""
    import torch
    pred_vol = pred_vol.clamp(eps, 1.0 - eps)
    return -(gt_vol * torch.log(pred_vol) + (1.0 - gt_vol) * torch.log(1.0 - pred_vol)).mean()


def _ce_loss_weighted(pred: "torch.Tensor", gt: "torch.Tensor",
                      w_pos: float = 50.0,
                      eps: float = 1e-7) -> "torch.Tensor":
    """
    Weighted binary cross-entropy — upweights fracture voxels.

    Standard CE is dominated by background (>99.9% of voxels).
    We multiply the fracture term by w_pos so each fracture voxel
    contributes w_pos× more than each background voxel.

    Option 1 from sampling strategy discussion.

    w_pos   : weight for fracture class (default 50, clipped inverse freq)
              A value of 50 means 1 fracture voxel = 50 background voxels.
              Compute as: min((1-f)/f, 100) where f = fracture fraction.
    """
    import torch
    pred = pred.clamp(eps, 1.0 - eps)
    return -(w_pos * gt * torch.log(pred) +
             (1.0 - gt) * torch.log(1.0 - pred)).mean()


def _focal_loss(pred: "torch.Tensor", gt: "torch.Tensor",
                gamma: float = 2.0,
                eps: float = 1e-7) -> "torch.Tensor":
    """
    Focal loss (Lin et al. 2017, RetinaNet).

    Automatically down-weights easy voxels — no manual class weight needed.
    The modulating factor (1-pt)^gamma reduces the loss for voxels where
    the model is already confident (e.g. easy background far from fracture).
    Only hard/uncertain voxels near the fracture boundary contribute strongly.

    Option 2 from sampling strategy discussion.

    L_focal = -sum( (1 - pt)^gamma * log(pt) )
    where pt = p if g=1 (fracture), pt = 1-p if g=0 (background)

    gamma : focusing parameter (default 2, standard value from Lin et al.)
            gamma=0 reduces to standard CE.
            gamma=2 down-weights easy examples by up to 100×.

    Reference:
        Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
    """
    import torch
    pred = pred.clamp(eps, 1.0 - eps)
    # pt = probability of the correct class
    pt   = pred * gt + (1.0 - pred) * (1.0 - gt)
    return -(((1.0 - pt) ** gamma) * torch.log(pt)).mean()


def _compute_pos_weight(gt: "torch.Tensor", w_max: float = 100.0) -> float:
    """
    Compute inverse-frequency weight for fracture class, clipped at w_max.
    Used for weighted CE (Option 1).

    gt : (H, W) or (N, H, W) binary ground truth
    Returns scalar float w_pos.
    """
    n_pos = gt.sum().item()
    n_tot = gt.numel()
    n_neg = n_tot - n_pos
    if n_pos == 0:
        return 1.0
    return min(n_neg / n_pos, w_max)


def _postprocess_3d(pred_vol_np: "np.ndarray",
                    min_component_voxels: int = 10,
                    close_gap_slices: int = 2) -> "np.ndarray":
    """
    3D post-processing for fracture predictions.

    Why needed: slice-by-slice SAM3 predictions have no inter-slice consistency.
    Common artefacts:
      - Isolated single-voxel or few-voxel false positives (noise)
      - Small blobs far from the main fracture
      - Gaps of 1-2 empty slices within a continuous fracture crack

    Steps:
      1. Remove connected components smaller than min_component_voxels.
         Default = 10 voxels — chosen to match the thinnest meaningful fracture:
         ~3 voxels thick × 3 voxels wide × 1 slice = 9 voxels.
         Set higher (e.g. 50) for aggressive noise removal if false positives dominate.

      2. Close small gaps across slices using binary dilation in Z direction only.
         Bridges fracture predictions interrupted by 1-2 empty slices.
         Handles multiple fractures correctly — dilation is per-region in XY space
         so spatially separate fractures don't accidentally merge.

    pred_vol_np         : (N, H, W) binary numpy array
    min_component_voxels: blobs smaller than this are removed (default 10)
    close_gap_slices    : number of slices to bridge (default 2)

    Returns cleaned (N, H, W) binary numpy array.
    """
    from scipy.ndimage import label, binary_dilation
    import numpy as np

    pred = pred_vol_np.astype(bool)

    # ── Step 1: Remove small components ──────────────────────────────────────
    labeled, n_comp = label(pred)
    cleaned = np.zeros_like(pred)
    for i in range(1, n_comp + 1):
        comp = labeled == i
        if comp.sum() >= min_component_voxels:
            cleaned |= comp

    # ── Step 2: Close gaps across slices (Z direction only) ──────────────────
    # Dilate in Z direction by close_gap_slices, then AND with original + dilation
    # This bridges short gaps without expanding in X/Y
    if close_gap_slices > 0:
        z_kernel = np.zeros((2 * close_gap_slices + 1, 1, 1), dtype=bool)
        z_kernel[:, 0, 0] = True
        dilated  = binary_dilation(cleaned, structure=z_kernel)
        # Only keep dilation where both neighbours agree (conservative bridging)
        # Re-label and keep components that grew from existing predictions
        cleaned  = dilated & (binary_dilation(cleaned, structure=z_kernel,
                                               iterations=close_gap_slices))

    return cleaned.astype(np.uint8)


def _sample_patches(img_np: "np.ndarray", gt_np: "np.ndarray",
                    patch_hw: tuple = (64, 64),
                    n_patches: int = 8,
                    fg_fraction: float = 0.33) -> list:
    """
    nnU-Net-style patch oversampling within a 2D slice.

    Instead of using the full slice (where fracture = <0.1% of voxels),
    we crop small patches — 33% centred on fracture voxels (foreground),
    67% randomly sampled anywhere (background + context).

    This matches nnU-Net's exact 33% foreground oversampling strategy.
    33% (not 50%) is deliberate — the model still sees mostly background
    so it learns not to over-predict, but fractures are seen often enough
    to get strong gradient signal. In a given epoch some fracture slices
    may not be sampled, but the ones that are give much stronger signal
    than full-slice training where fracture = 0.01% of voxels.

    Option 3 from sampling strategy discussion.

    img_np      : (H, W, C) float image slice
    gt_np       : (H, W) binary ground truth
    patch_hw    : (ph, pw) patch size in pixels (default 64×64)
    n_patches   : total patches to extract per slice (default 8)
    fg_fraction : fraction of patches centred on fracture (default 0.5)

    Returns list of dicts with keys 'image' (ph,pw,C) and 'gt' (ph,pw).
    """
    import numpy as np
    H, W = gt_np.shape
    ph, pw = patch_hw
    patches = []

    # Find fracture voxel locations
    fg_coords = np.argwhere(gt_np > 0)  # (N_fg, 2) — row, col indices

    n_fg = max(1, int(n_patches * fg_fraction))
    n_bg = n_patches - n_fg

    def extract(cy, cx):
        """Extract patch centred at (cy, cx), clamped to image bounds."""
        y0 = max(0, min(H - ph, cy - ph // 2))
        x0 = max(0, min(W - pw, cx - pw // 2))
        return {
            "image": img_np[y0:y0+ph, x0:x0+pw, :],
            "gt":    gt_np[y0:y0+ph, x0:x0+pw].astype(np.float32),
        }

    # Foreground patches — centred on random fracture voxels
    if len(fg_coords) > 0:
        chosen = fg_coords[np.random.choice(len(fg_coords), n_fg, replace=True)]
        for cy, cx in chosen:
            patches.append(extract(int(cy), int(cx)))
    else:
        n_bg += n_fg  # no fracture voxels — fall back to random

    # Background patches — random locations
    for _ in range(n_bg):
        cy = np.random.randint(ph // 2, H - ph // 2 + 1)
        cx = np.random.randint(pw // 2, W - pw // 2 + 1)
        patches.append(extract(cy, cx))

    return patches


def _build_sam3_lora(rank: int = 4, device: str = "cpu", use_medsam3: bool = False,
                     use_text_prompt: bool = False):
    """
    Load SAM3 + inject true LoRA matrices into the ViT backbone.

    Architecture follows Sompote SAM3_LoRA (github.com/Sompote/SAM3_LoRA):
        Original weights W frozen — never updated
        Low-rank matrices A, B injected into Q, K, V, fc1, fc2 of every
        attention block:  W_effective = W + (alpha/rank) * B @ A
        Segmentation head: lightweight conv stack on top of backbone features

    This is true LoRA (Hu et al. 2022) — the backbone adapts its internal
    representations to CT fracture data through A and B, while W is frozen.
    Feature caching is NOT used because features change every epoch as A, B update.

    Parameter count:
        Original backbone W:  840M  (frozen, no gradients)
        LoRA matrices A+B:    ~2-8M depending on rank  (trainable)
        Seg head:             ~40K  (trainable)
        Total trainable:      ~0.3% of backbone at rank=4

    References:
        Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
        Sompote (2024). SAM3_LoRA. github.com/Sompote/SAM3_LoRA

    Returns (model, processor, seg_head, trainable_params).
    """
    import torch
    import torch.nn as nn
    import math

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        raise RuntimeError(
            "SAM3 not found.\n"
            "Run with base Python: /opt/homebrew/Caskroom/miniconda/base/bin/python3 fracture_pipeline.py train_sam\n"
            "Or: conda activate base"
        )

    # ── LoRA linear layer ─────────────────────────────────────────────────────
    class LoRALinear(nn.Module):
        """
        Wraps an existing nn.Linear with LoRA low-rank adaptation.

        Forward: W_effective = W + (alpha/rank) * B @ A
        where W is the original frozen weight, A and B are trained.

        A is initialised with Kaiming uniform (standard), B with zeros —
        so at epoch 0 the LoRA contribution is exactly zero and training
        starts from the pretrained model. This is the standard LoRA init
        from Hu et al. 2022.
        """
        def __init__(self, linear: nn.Linear, rank: int, alpha: float):
            super().__init__()
            self.linear   = linear          # original frozen layer
            self.rank     = rank
            self.alpha    = alpha
            in_f  = linear.in_features
            out_f = linear.out_features

            # LoRA matrices: A projects down to rank, B projects back up
            self.lora_A = nn.Linear(in_f,  rank,  bias=False)
            self.lora_B = nn.Linear(rank,  out_f, bias=False)

            # Standard LoRA initialisation
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            # Freeze original weights — only A and B train
            for p in self.linear.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.linear(x) + (self.alpha / self.rank) * self.lora_B(self.lora_A(x))

    # ── Load SAM3 backbone ────────────────────────────────────────────────────
    # Patch SAM3 position encoding to use correct device (CPU when no GPU available)
    import sam3.model.position_encoding as _pe_mod
    _orig_pe_init = _pe_mod.PositionEmbeddingSine.__init__
    def _patched_pe_init(self, *args, **kwargs):
        # Temporarily redirect cuda tensors to cpu if cuda not available
        import torch
        _orig_device = torch.zeros.__defaults__
        _orig_pe_init(self, *args, **kwargs)
    # Simpler fix: patch torch.zeros to not use cuda in position encoding
    import sam3.model.position_encoding as _pe
    _orig_forward = _pe.PositionEmbeddingSine.forward
    def _safe_forward(self, tensor_list):
        import torch
        device = next(iter(tensor_list.tensors if hasattr(tensor_list, 'tensors') else [tensor_list]), None)
        if device is not None and hasattr(device, 'device'):
            self.not_mask = self.not_mask.to(device.device) if hasattr(self, 'not_mask') else self.not_mask
        return _orig_forward(self, tensor_list)

    print("  Loading SAM3 backbone...")
    # Force CPU-safe loading by temporarily making CUDA unavailable to SAM3
    import os
    _orig_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        model = build_sam3_image_model(device="cpu", eval_mode=False, load_from_HF=True)
    except Exception as e:
        print(f"  HF download failed ({e}), loading without pretrained weights...")
        try:
            model = build_sam3_image_model(device="cpu", eval_mode=False, load_from_HF=False)
        except Exception as e2:
            print(f"  SAM3 load failed ({e2})")
            raise RuntimeError(f"Cannot load SAM3: {e2}")
    finally:
        if device == "cpu":
            if _orig_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cuda_visible

    # Freeze ALL parameters first
    for p in model.parameters():
        p.requires_grad = False

    # ── Optionally load MedSAM3 LoRA weights ─────────────────────────────────
    # MedSAM3 weights are LoRA weights trained on 658K medical images across
    # CT, MRI, X-ray etc. Loading them gives a much better medical starting
    # point than raw SAM3 (natural images only).
    # Source: https://huggingface.co/lal-Joey/MedSAM3_v1
    if use_medsam3:
        print("  Loading MedSAM3 medical LoRA weights...")
        try:
            from huggingface_hub import hf_hub_download
            import os
            hf_home = os.environ.get("HF_HOME",
                      os.path.expanduser("~/.cache/huggingface"))
            cache_dir = os.path.join(hf_home, "hub")
            # Try to find the weights file — common filenames
            for fname in ["best_lora_weights.pt", "last_lora_weights.pt",
                          "pytorch_model.bin", "model.safetensors",
                          "lora_weights.pt"]:
                try:
                    weight_path = hf_hub_download(
                        repo_id="lal-Joey/MedSAM3_v1",
                        filename=fname,
                        cache_dir=cache_dir,
                    )
                    medsam3_weights = torch.load(weight_path, map_location="cpu")
                    # Load with strict=False — only matching keys loaded
                    missing, unexpected = model.load_state_dict(
                        medsam3_weights, strict=False
                    )
                    n_loaded = len(medsam3_weights) - len(unexpected)
                    print(f"  ✓ MedSAM3 weights loaded: {n_loaded} layers "
                          f"({len(missing)} missing, {len(unexpected)} unexpected)")
                    break
                except Exception:
                    continue
            else:
                print("  ⚠️  MedSAM3 weights not found on HuggingFace — "
                      "using raw SAM3 weights instead.")
                print("  Download manually from: "
                      "https://huggingface.co/lal-Joey/MedSAM3_v1")
        except ImportError:
            print("  ⚠️  huggingface_hub not installed — "
                  "pip install huggingface_hub")
        except Exception as e:
            print(f"  ⚠️  MedSAM3 load failed ({e}) — using raw SAM3")

    # ── Inject LoRA via apply_lora_to_model ───────────────────────────────────
    # Uses the proven LoRALinear implementation from lora_layers.py.
    #
    # SAM3 actual module names (from inspection):
    #   - Vision backbone:  backbone.vision_backbone.trunk.blocks.N.attn.qkv (fused!)
    #                       backbone.vision_backbone.trunk.blocks.N.attn.proj
    #                       backbone.vision_backbone.trunk.blocks.N.mlp.fc1
    #                       backbone.vision_backbone.trunk.blocks.N.mlp.fc2
    #
    # Note: SAM3 uses FUSED qkv (1024→3072), not separate q_proj/k_proj/v_proj.
    # The old code looked for q_proj which doesn't exist → 0 LoRA layers injected.
    try:
        from lora_layers import LoRAConfig, apply_lora_to_model
    except ImportError as e:
        raise ImportError(
            "lora_layers.py not found. Make sure it is in the same dir as "
            "fracture_pipeline.py."
        ) from e

    alpha = float(rank * 2)
    lora_cfg = LoRAConfig(
        rank=rank,
        alpha=int(alpha),
        dropout=0.0,
        target_modules=["qkv", "proj", "fc1", "fc2", "out_proj", "linear1", "linear2", "c_fc", "c_proj"],
        apply_to_vision_backbone=True,
        apply_to_text=use_text_prompt,    # enable for text prompt mode
        apply_to_geometry=False,
        apply_to_head=use_text_prompt,    # enable mask decoder LoRA when using prompts
    )
    model = apply_lora_to_model(model, lora_cfg, verbose=True)

    n_injected = sum(1 for n, _ in model.named_modules() if "lora_A" in n)
    extra_msg  = ""
    if use_text_prompt:
        extra_msg = " + text encoder + mask decoder (head)"
    print(f"  ✓ LoRA active on vision backbone{extra_msg} (rank={rank}, alpha={int(alpha)})")

    processor = Sam3Processor(model)

    model = model.to(device)


    # ── Segmentation head (used when use_medsam3=False) ──────────────────────
    # Fuller conv head: 256→64→16→1 (~40K params)
    # Used when no text prompt — learns to map backbone features to fracture mask.
    # When use_medsam3=True this is bypassed (mask decoder used directly).
    seg_head = nn.Sequential(
        nn.Conv2d(256, 64, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 1, kernel_size=1, bias=True),
        nn.Sigmoid()
    ).to(device)

    # ── Collect trainable parameters ──────────────────────────────────────────
    # LoRA A,B matrices are auto-marked requires_grad=True by apply_lora_to_model.
    # Just collect all params with requires_grad=True (LoRA + seg head).
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(seg_head.parameters())

    total_sam3 = sum(p.numel() for p in model.parameters())
    n_lora     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head     = sum(p.numel() for p in seg_head.parameters())

    print(f"  SAM3 backbone:     {total_sam3:,} params total (frozen except LoRA)")
    print(f"  LoRA params:       {n_lora:,} (rank={rank}, vision backbone)")
    print(f"  Seg head:          {n_head:,} (Conv2d 256→1)")
    print(f"  Total trainable:   {n_lora + n_head:,}  "
          f"({100*(n_lora+n_head)/max(total_sam3,1):.3f}% of backbone)")

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
                   run_validation: bool = False,
                   loss_type: str = "dice_3d_ce_boundary",
                   input_mode: str = "2.5d",
                   postprocess: bool = False,
                   sampling: str = "weighted_ce",
                   ce_weight: float = 50.0,
                   patch_size: int = 64,
                   n_patches: int = 8,
                   fg_fraction: float = 0.33,
                   patch_ce_weight: float = 0.0,
                   exp_name: str = None,
                   use_medsam3: bool = False,
                   use_text_prompt: bool = False) -> None:
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

    Recommended training configurations:

        Laptop / baseline:
            --loss dice_3d_ce_boundary --sampling weighted_ce
            Full slice, 3D Dice + weighted CE + boundary loss.
            Weighted CE upweights fracture voxels by inverse frequency
            so background does not dominate gradient. Boundary loss
            annealed in from epoch 25. All losses computed over full
            assembled 3D volume.

        HPC ablation A (no sampling):
            --loss dice_3d_ce_boundary --sampling weighted_ce --validate
            Same as laptop but with LoRA and validation each epoch.

        HPC ablation B (patch sampling):
            --loss dice --sampling patch --patch_size 96 --n_patches 8 --fg_fraction 0.33
            33%% fg patch oversampling (nnU-Net style). Strongest class
            balance but 3D Dice and boundary loss are not used (patches
            cannot be assembled into a full volume).

        HPC ablation C (focal loss):
            --loss dice_3d_ce_boundary --sampling focal --validate
            Focal loss replaces weighted CE. No manual weight tuning.

    Loss options (--loss flag):
        dice                — plain per-slice Dice (legacy baseline)
        dice_ce             — per-slice Dice + CE
        dice_ce_boundary    — per-slice Dice + CE + 2D boundary loss
        dice_3d             — true 3D Dice over full assembled volume
        dice_3d_ce          — 3D Dice + 3D CE
        dice_3d_ce_boundary — 3D Dice + 3D CE + 3D boundary loss  [DEFAULT]

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

    # Memory optimization for P100 16GB
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # MedSAM3 always uses text prompt
    if use_medsam3:
        use_text_prompt = True

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
    # Auto-generate experiment name from key hyperparameters if not provided
    if exp_name is None:
        base = "medsam3" if use_medsam3 else "sam3"
        prompt_tag = "_prompt" if use_text_prompt else ""
        exp_name = (
            f"{base}{prompt_tag}"
            f"_s{sampling}"
            f"_r{lora_rank}"
            f"_ctx{context}"
            f"_lr{lr:.0e}"
            f"_{'val' if run_validation else 'noval'}"
        )
        if sampling == "patch":
            exp_name += f"_p{patch_size}_n{n_patches}_fg{fg_fraction}"
            if patch_ce_weight > 0:
                exp_name += f"_cew{int(patch_ce_weight)}"
        elif sampling == "weighted_ce":
            exp_name += f"_w{int(ce_weight)}"
    sam_fold_dir = BASE_DIR / "sam_results" / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}_fold{fold}_{exp_name}"
    pred_dir     = sam_fold_dir / "predictions"
    sam_fold_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Experiment: {exp_name}")
    print(f"  Output dir: {sam_fold_dir}")

    # ── Build model ───────────────────────────────────────────────────────
    print("\n  Building SAM3 + LoRA head...")
    try:
        model, processor, seg_head, trainable_params = _build_sam3_lora(
            rank=lora_rank, device=device, use_medsam3=use_medsam3,
            use_text_prompt=use_text_prompt,
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
        # input_mode='single': context=0 means no neighbours stacked (HPC v3)
        # input_mode='2.5d':   context=N stacks ±N neighbours (default)
        effective_context = 0 if input_mode == "single" else context
        slices = _ct_to_sam3_slices(ct_path, mask_path,
                                     context=effective_context, target_hw=target_hw)
        all_train_slices.extend(slices)
        print(f"    {pid}: {len(slices)} fracture slices")

    if not all_train_slices:
        print("  ✗ No training slices found.")
        return
    print(f"  Total training slices: {len(all_train_slices)}")

    # ── Pre-load GT volumes and voxel spacing ──────────────────────────────
    # With true LoRA, features change every epoch so NO caching.
    # Instead we load CT slices fresh each epoch via _ct_to_sam3_slices.
    import nibabel as nib
    from PIL import Image as PILImage
    import torch.nn.functional as F_nn
    from skimage.transform import resize as sk_resize

    gt_cache   = {}   # pid -> (n_slices, H, W) tensor
    zooms_cache = {}  # pid -> voxel spacing tuple

    print("\n  Loading GT volumes...")
    for pid in train_pids:
        mask_path = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
        if not mask_path.exists():
            continue
        mask_nib  = nib.load(str(mask_path))
        mask_vol  = np.transpose(mask_nib.get_fdata().astype(np.float32), (1, 0, 2))
        n_sl      = mask_vol.shape[0]
        gt_t      = torch.zeros(n_sl, *target_hw)
        for si in range(n_sl):
            gt_r = sk_resize(mask_vol[si], target_hw, order=0,
                             preserve_range=True).astype(np.float32)
            gt_t[si] = torch.from_numpy(gt_r)
        gt_cache[pid]    = gt_t
        zooms_cache[pid] = mask_nib.header.get_zooms()
        print(f"    {pid}: GT loaded  ({n_sl} slices)")

    print(f"  No feature caching — LoRA weights change every epoch (correct behaviour).")
    print(f"  Training with full backbone forward pass per slice...")

    # ── Training loop ────────────────────────────────────────────────────────
    log       = []
    best_loss = float("inf")
    head_device = next(seg_head.parameters()).device
    n_channels  = 1 if input_mode == "single" else (2 * context + 1)

    # Loss flags — computed once outside epoch loop
    use_boundary = loss_type in ("dice_ce_boundary", "dice_3d_ce_boundary")
    use_3d       = loss_type in ("dice_3d", "dice_3d_ce", "dice_3d_ce_boundary")
    use_ce       = loss_type in ("dice_ce", "dice_ce_boundary",
                                 "dice_3d_ce", "dice_3d_ce_boundary")

    for epoch in range(n_epochs):
        model.train()
        seg_head.train()        # LoRA matrices need grad — backbone in train mode
        model.train()
        seg_head.train()
        epoch_dice_losses  = []
        epoch_ce_losses    = []
        epoch_total_losses = []

        # Annealed lambda: boundary loss 0→1 over first 50% of training
        lam_boundary = min(1.0, epoch / max(1, n_epochs * 0.5)) if use_boundary else 0.0

        for pid in train_pids:
            if pid not in gt_cache:
                continue

            ct_path     = CONVERTED_DIR / f"{pid}_ct.nii.gz"
            gt_vol_full = gt_cache[pid]
            zooms       = zooms_cache[pid]

            # Load fracture slices for this patient (only slices with GT fracture)
            effective_context = 0 if input_mode == "single" else context
            pid_slices  = _ct_to_sam3_slices(ct_path, CONVERTED_DIR / f"{pid}_fracture.nii.gz",
                                              context=effective_context, target_hw=target_hw)
            if not pid_slices:
                continue

            optimizer.zero_grad()

            pred_slices = []
            gt_slices   = []
            loss_sum_2d  = torch.tensor(0.0, device=head_device, requires_grad=False)
            dice_sum_2d  = torch.tensor(0.0, device=head_device, requires_grad=False)
            ce_sum_2d    = torch.tensor(0.0, device=head_device, requires_grad=False)
            n_valid      = 0

            for sl in pid_slices:
                try:
                    idx    = sl["slice_idx"]
                    img_np = sl["image"]   # (H, W, C)
                    gt_np  = gt_vol_full[idx].numpy()  # (H, W)

                    # ── Build list of (img_patch, gt_patch) units to process ──
                    # full_slice: one unit = entire slice
                    # patch:      multiple small patches per slice, 50% fg-centred
                    if sampling == "patch":
                        units = _sample_patches(
                            img_np, gt_np,
                            patch_hw=(patch_size, patch_size),
                            n_patches=n_patches,
                            fg_fraction=fg_fraction,
                        )
                    else:
                        units = [{"image": img_np, "gt": gt_np.astype(np.float32)}]

                    for unit in units:
                        u_img = unit["image"]
                        u_gt  = torch.from_numpy(unit["gt"]).to(head_device)

                        # Ensure 3-channel RGB for SAM3 processor
                        if u_img.shape[2] != 3:
                            repeats = max(1, 3 // u_img.shape[2])
                            u_img   = np.concatenate([u_img] * repeats, axis=2)[:, :, :3]

                        img_uint8 = (u_img * 255).clip(0, 255).astype(np.uint8)
                        pil_img   = PILImage.fromarray(img_uint8, mode="RGB")

                        # Full forward pass through LoRA-adapted backbone + text prompt
                        inf_state  = processor.set_image(pil_img)
                        # Extract backbone features → sigmoid for binary prediction
                        bfeats    = inf_state["backbone_out"]["vision_features"]
                        bfeats_r  = F_nn.interpolate(
                            bfeats.float(), size=(u_img.shape[0], u_img.shape[1]),
                            mode="bilinear", align_corners=False).to(head_device)
                        pred_2d   = seg_head(bfeats_r).squeeze()
                        del inf_state, bfeats, bfeats_r
                        if device == "cuda" and n_valid % 5 == 0:
                            torch.cuda.empty_cache()

                        if use_3d and sampling != "patch":
                            # Only collect full slices for 3D volume assembly
                            # Patch mode always uses per-patch loss
                            pred_slices.append(pred_2d)
                            gt_slices.append(u_gt)
                        else:
                            # Per-unit loss computation
                            smooth  = 1e-5
                            inter   = (pred_2d * u_gt).sum()
                            dice_sl = 1.0 - (2.0 * inter + smooth) / (
                                      pred_2d.sum() + u_gt.sum() + smooth)

                            if sampling == "weighted_ce" and use_ce:
                                # Weighted CE over full slice
                                w_pos   = _compute_pos_weight(u_gt, w_max=ce_weight)
                                sl_loss = dice_sl + _ce_loss_weighted(pred_2d, u_gt, w_pos=w_pos)
                            elif sampling == "focal":
                                # Focal loss replaces CE entirely
                                sl_loss = dice_sl + _focal_loss(pred_2d, u_gt, gamma=2.0)
                            elif sampling == "patch" and patch_ce_weight > 0 and use_ce:
                                # Patch + weighted CE — mirrors nnU-Net strategy
                                w_pos   = _compute_pos_weight(u_gt, w_max=patch_ce_weight)
                                sl_loss = dice_sl + _ce_loss_weighted(pred_2d, u_gt, w_pos=w_pos)
                            else:
                                # full_slice or patch without CE weighting: plain CE
                                sl_loss = dice_sl
                                if use_ce:
                                    sl_loss = sl_loss + _ce_loss_2d(pred_2d, u_gt)

                            if use_boundary and sampling != "patch":
                                b_sl    = _boundary_loss_2d(
                                    pred_2d, unit["gt"],
                                    spacing=(float(zooms[0]), float(zooms[2])))
                                sl_loss = sl_loss + lam_boundary * b_sl

                            loss_sum_2d = loss_sum_2d + sl_loss
                            # Track dice and ce components separately for reporting
                            dice_sum_2d = dice_sum_2d + dice_sl.detach()
                            ce_sum_2d   = ce_sum_2d + (sl_loss - dice_sl).detach()
                            del pred_2d, u_gt

                    n_valid += 1
                    if n_valid % 10 == 0:
                        import gc; gc.collect()
                        if hasattr(torch, "mps") and torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                except Exception as e:
                    if epoch == 0:
                        print(f"    Forward pass error (slice {sl['slice_idx']}): {e}")
                    continue

            if n_valid == 0:
                continue

            # ── Compute loss ─────────────────────────────────────────────────
            if use_3d and pred_slices:
                pred_vol_t = torch.stack(pred_slices, dim=0)
                gt_vol_t   = torch.stack(gt_slices,   dim=0)
                smooth     = 1e-5
                inter_3d   = (pred_vol_t * gt_vol_t).sum()
                loss       = 1.0 - (2.0 * inter_3d + smooth) / (
                             pred_vol_t.sum() + gt_vol_t.sum() + smooth)

                # CE term — use appropriate variant based on --sampling
                if use_ce:
                    if sampling == "weighted_ce":
                        # Option 1: weighted CE over full 3D volume
                        w_pos = _compute_pos_weight(gt_vol_t, w_max=ce_weight)
                        loss  = loss + _ce_loss_weighted(pred_vol_t, gt_vol_t, w_pos=w_pos)
                    elif sampling == "focal":
                        # Option 2: focal loss over full 3D volume
                        loss = loss + _focal_loss(pred_vol_t, gt_vol_t, gamma=2.0)
                    else:
                        # full_slice default: plain 3D CE
                        loss = loss + _ce_loss_3d(pred_vol_t, gt_vol_t)

                if use_boundary:
                    loss = loss + lam_boundary * _boundary_loss_3d(
                        pred_vol_t, gt_vol_t.cpu().numpy(),
                        spacing=(float(zooms[1]), float(zooms[0]), float(zooms[2])))
                del pred_vol_t, gt_vol_t
            else:
                loss = loss_sum_2d / n_valid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_dice_losses.append((dice_sum_2d / n_valid).item())
            epoch_ce_losses.append((ce_sum_2d / n_valid).item())
            epoch_total_losses.append(loss.item())

        mean_dice_l  = float(np.mean(epoch_dice_losses))  if epoch_dice_losses  else float("nan")
        mean_ce_l    = float(np.mean(epoch_ce_losses))    if epoch_ce_losses    else float("nan")
        mean_loss    = float(np.mean(epoch_total_losses)) if epoch_total_losses else float("nan")
        # Dice score = 1 - dice_loss
        train_dice_score = 1.0 - mean_dice_l if not np.isnan(mean_dice_l) else float("nan")

        # ── Validation on true held-out patient (HPC mode only) ──────────
        # Skipped by default on laptop to avoid OOM. Enable with --validate.
        if run_validation:
            model.eval()
            seg_head.eval()
            val_losses = []
            with torch.no_grad():
                for pid in val_pids:
                    ct_path   = CONVERTED_DIR / f"{pid}_ct.nii.gz"
                    mask_path = CONVERTED_DIR / f"{pid}_fracture.nii.gz"
                    if not ct_path.exists():
                        continue

                    # Load val GT if not already cached
                    if pid not in gt_cache:
                        mask_nib = nib.load(str(mask_path))
                        mask_vol = np.transpose(mask_nib.get_fdata().astype(np.float32), (1, 0, 2))
                        n_sl = mask_vol.shape[0]
                        gt_t = torch.zeros(n_sl, *target_hw)
                        from skimage.transform import resize as sk_resize
                        for si in range(n_sl):
                            gt_r = sk_resize(mask_vol[si], target_hw, order=0, preserve_range=True).astype(np.float32)
                            gt_t[si] = torch.from_numpy(gt_r)
                        gt_cache[pid] = gt_t

                    val_gt = gt_cache[pid]
                    # Use fracture slices only for validation
                    effective_context_val = 0 if input_mode == "single" else context
                    val_slices = _ct_to_sam3_slices(ct_path, mask_path,
                                                    context=effective_context_val,
                                                    target_hw=target_hw)

                    for sl in val_slices:
                        idx    = sl["slice_idx"]
                        img_np = sl["image"]
                        if img_np.shape[2] != 3:
                            repeats = max(1, 3 // img_np.shape[2])
                            img_np  = np.concatenate([img_np] * repeats, axis=2)[:, :, :3]
                        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
                        pil_img   = PILImage.fromarray(img_uint8, mode="RGB")
                        try:
                            # Check GT index is valid
                            if idx >= val_gt.shape[0]:
                                continue
                            inf_state  = processor.set_image(pil_img)
                            if use_text_prompt:
                                out_v      = processor.set_text_prompt(
                                    state=inf_state, prompt="pelvic fracture"
                                )
                                if out_v is None:
                                    del inf_state
                                    continue
                                masks_v    = (out_v.get("masks", None) if isinstance(out_v, dict)
                                              else getattr(out_v, "masks", None))
                                scores_v   = (out_v.get("scores", None) if isinstance(out_v, dict)
                                              else getattr(out_v, "scores", None))
                                if masks_v is None or len(masks_v) == 0:
                                    del inf_state, out_v
                                    continue
                                best_v     = int(scores_v.argmax()) if scores_v is not None else 0
                                mask_v_np  = masks_v[best_v].squeeze().astype(np.float32)
                                from skimage.transform import resize as sk_resize
                                mask_v_r   = sk_resize(mask_v_np, target_hw, order=1, preserve_range=True)
                                pred_2d    = torch.from_numpy(mask_v_r).float().to(head_device)
                                del inf_state, out_v, masks_v
                            else:
                                bfeats_v   = inf_state["backbone_out"]["vision_features"]
                                bfeats_r_v = F_nn.interpolate(
                                    bfeats_v.float(), size=target_hw,
                                    mode="bilinear", align_corners=False).to(head_device)
                                pred_2d    = seg_head(bfeats_r_v).squeeze()
                                del inf_state, bfeats_v, bfeats_r_v
                            gt_2d      = val_gt[idx].to(head_device)
                            smooth     = 1e-5
                            inter      = (pred_2d * gt_2d).sum()
                            val_dice   = 1.0 - (2.0 * inter + smooth) / (pred_2d.sum() + gt_2d.sum() + smooth)
                            val_losses.append(val_dice.item())
                            del pred_2d, gt_2d
                        except Exception as e:
                            if epoch == 0 and len(val_losses) == 0:
                                print(f"    ⚠️  Val forward pass error (slice {idx}): {e}")
                            continue

        if run_validation:
            model.train()
            seg_head.train()
            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            val_dice_score = 1.0 - mean_val_loss if not np.isnan(mean_val_loss) else float("nan")
            log.append({"epoch": epoch,
                        "dice_loss": mean_dice_l, "ce_loss": mean_ce_l, "total_loss": mean_loss,
                        "train_dice": train_dice_score,
                        "val_dice_loss": mean_val_loss, "val_dice": val_dice_score,
                        "lambda_boundary": round(lam_boundary, 4)})
            val_str = f"  Val_Dice={val_dice_score:.4f}" if not np.isnan(val_dice_score) else "  Val_Dice=nan"
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"Dice={train_dice_score:.4f}  "
                  f"CE={mean_ce_l:.4f}  "
                  f"Total={mean_loss:.4f}{val_str}  λ={lam_boundary:.2f}")
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                ckpt_path = sam_fold_dir / "best_lora.pt"
                torch.save({name: p for name, p in model.named_parameters() if p.requires_grad}, ckpt_path)
                print(f"    ✓ Saved best checkpoint (val_loss={best_loss:.4f})")
        else:
            log.append({"epoch": epoch,
                        "dice_loss": mean_dice_l, "ce_loss": mean_ce_l, "total_loss": mean_loss,
                        "train_dice": train_dice_score,
                           "lambda_boundary": round(lam_boundary, 4)})
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"Dice={train_dice_score:.4f}  "
                  f"CE={mean_ce_l:.4f}  "
                  f"Total={mean_loss:.4f}  λ={lam_boundary:.2f}")
            if mean_loss < best_loss:
                best_loss = mean_loss
                ckpt_path = sam_fold_dir / "best_lora.pt"
                torch.save({name: p for name, p in model.named_parameters() if p.requires_grad}, ckpt_path)
                print(f"    ✓ Saved best checkpoint (loss={best_loss:.4f})")

    # Save training log
    with open(sam_fold_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # ── Predict held-out patient ──────────────────────────────────────────
    print(f"\n  Predicting held-out patient: {val_pids}")
    model.eval()
    model.eval()

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

                    inf_state  = processor.set_image(pil_img)
                    if use_text_prompt:
                        out_i      = processor.set_text_prompt(
                            state=inf_state, prompt="pelvic fracture"
                        )
                        if out_i is not None:
                            masks_i    = (out_i.get("masks", None) if isinstance(out_i, dict)
                                          else getattr(out_i, "masks", None))
                            scores_i   = (out_i.get("scores", None) if isinstance(out_i, dict)
                                          else getattr(out_i, "scores", None))
                            if masks_i is not None and len(masks_i) > 0:
                                best_i     = int(scores_i.argmax()) if scores_i is not None else 0
                                mask_i_np  = masks_i[best_i].squeeze().astype(np.float32)
                                mask_r     = sk_resize(mask_i_np, target_hw, order=1, preserve_range=True)
                                pred_orig  = sk_resize(mask_r, (orig_shape[0], orig_shape[2]),
                                                       order=1, preserve_range=True)
                                pred_vol[i] = pred_orig
                        del inf_state, out_i
                    else:
                        bfeats_i   = inf_state["backbone_out"]["vision_features"]
                        bfeats_r_i = F_nn.interpolate(
                            bfeats_i.float(), size=target_hw,
                            mode="bilinear", align_corners=False).to(next(seg_head.parameters()).device)
                        pred_map   = seg_head(bfeats_r_i).squeeze()
                        mask_r     = pred_map.detach().cpu().numpy()
                        pred_orig  = sk_resize(mask_r, (orig_shape[0], orig_shape[2]),
                                               order=1, preserve_range=True)
                        pred_vol[i] = pred_orig
                        del inf_state, bfeats_i, bfeats_r_i

                except Exception:
                    pass

        # Threshold
        pred_binary_raw = (pred_vol > 0.5).astype(np.uint8)  # (N_slices, X, Z)

        # Optional 3D post-processing: remove noise + bridge gaps
        if postprocess:
            pred_binary_raw = _postprocess_3d(
                pred_binary_raw,
                min_component_voxels=10,  # ~3×3×1 voxels = smallest real fracture tip
                close_gap_slices=2        # bridge gaps up to 2 empty slices
            )
            n_before = (pred_vol > 0.5).sum()
            n_after  = pred_binary_raw.sum()
            print(f"  Post-processing: {n_before} → {n_after} voxels "
                  f"({100*(n_after-n_before)/max(n_before,1):+.1f}%)")

        pred_binary = np.transpose(pred_binary_raw, (1, 0, 2)).astype(np.uint8)
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
        evaluate_all_folds(splits, model=args.model, eval_set=args.eval_set)

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
            loss_type=args.loss,
            input_mode=args.input_mode,
            postprocess=args.postprocess,
            sampling=args.sampling,
            ce_weight=args.ce_weight,
            patch_size=args.patch_size,
            n_patches=args.n_patches,
            fg_fraction=args.fg_fraction,
            patch_ce_weight=args.patch_ce_weight,
            exp_name=args.exp_name,
            use_medsam3=args.use_medsam3,
            use_text_prompt=args.use_text_prompt,
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

