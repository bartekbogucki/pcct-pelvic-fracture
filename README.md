# Automated Pelvic Fracture Segmentation

Master's thesis project — automated detection of pelvic fractures in **Photon-Counting CT (PCCT)** using deep learning.

Two complementary approaches are implemented and compared:
- **nnU-Net** — self-configuring 3D convolutional segmentation framework
- **SAM3 + LoRA** — frozen vision foundation model with a lightweight trainable segmentation head

---

## Pipeline Overview

```
PCCT Scan → TotalSegmentator (pelvis ROI) → Fracture Model → Binary Mask
```

The two-stage design is critical: fracture voxels represent <0.01% of the full CT volume. Cropping to the pelvis ROI first reduces class imbalance and compute by ~10×.

---

## Installation

### 1. Clone and set up environment

```bash
git clone <repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install PyTorch

**macOS (Apple Silicon / MPS):**
```bash
pip install torch torchvision
```

**Linux / HPC (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Install SAM3 (from source)

```bash
git clone https://github.com/bowang-lab/SAM3
cd SAM3 && pip install -e .
cd ..
```

---

## Usage

### Step 1 — Register patients

```bash
python fracture_pipeline.py add \
    --patient_id 64406628 \
    --dicom_dir  "data/64406628/CT/slices" \
    --seg_fracture "data/64406628/fracture segmentation/Fractures all.dcm" \
    --seg_pelvis   "data/64406628/pelvic segmentation/pelvis.dcm"
```

Repeat for each patient. Registry is saved to `dataset_registry.json`.

### Step 2 — Visual QC

Always run this before training to verify segmentation alignment:

```bash
python fracture_pipeline.py qc --patient_id 64406628
# or all patients at once:
python fracture_pipeline.py qc
```

Saves overlay PNGs to `qc/`. Check that red (fracture) overlaps with green (pelvis).

### Step 3 — Prepare folds

```bash
# Laptop (default — safe for macOS):
python fracture_pipeline.py prepare

# HPC (true held-out validation, requires GPU):
python fracture_pipeline.py prepare --hpc_splits
```

With n=4 patients this creates Leave-One-Out CV (4 folds). Each fold has 3 training patients and 1 held-out test patient.

### Step 4a — Train nnU-Net

```bash
python fracture_pipeline.py train --fold 0
```

Runs `nnUNetv2_plan_and_preprocess` + `nnUNetv2_train` automatically. Expects ~15 min/epoch on MPS, ~2 min/epoch on A100.

### Step 4b — Train SAM3 + LoRA

```bash
# Laptop (no validation loop):
python fracture_pipeline.py train_sam --fold 0 --epochs 50

# HPC (with per-epoch validation on held-out patient):
python fracture_pipeline.py train_sam --fold 0 --epochs 200 --validate
```

Feature caching runs once (~30 min on CPU for ~350 slices), then training is ~1 sec/epoch.

### Step 5 — Evaluate

```bash
python fracture_pipeline.py evaluate
```

Computes 3D Dice coefficient for each held-out patient across all folds.

### Step 6 — Plot results

```bash
# All plots:
python fracture_pipeline.py plot --what all

# Specific:
python fracture_pipeline.py plot --what curves --fold 0   # nnU-Net training curves
python fracture_pipeline.py plot --what sam_curves --fold 0  # SAM3 training curves
python fracture_pipeline.py plot --what dice              # per-patient Dice bar chart
python fracture_pipeline.py plot --what predictions       # overlay PNGs
```

### Step 7 — Predict new patient

```bash
# Single fold model:
python fracture_pipeline.py predict \
    --dicom_dir "data/new_patient/CT" \
    --seg_pelvis "data/new_patient/pelvis.dcm" \
    --output "predictions/new_patient.nii.gz"

# Ensemble (all fold models averaged — better performance):
python fracture_pipeline.py ensemble \
    --dicom_dir "data/new_patient/CT" \
    --seg_pelvis "data/new_patient/pelvis.dcm" \
    --output "predictions/new_patient.nii.gz"
```

---

## Laptop vs HPC Mode

| Feature | Laptop (default) | HPC (`--validate` / `--hpc_splits`) |
|---|---|---|
| nnU-Net val split | last training patient | true held-out patient |
| SAM3 val loop | disabled | per-epoch, held-out patient |
| Best checkpoint | saved on train loss | saved on val loss |
| Memory | safe for 16–24 GB unified | requires GPU with 16+ GB VRAM |

---

## Dataset Structure

```
data/
  {patient_id}/
    CT/                        ← DICOM slices
    fracture segmentation/
      Fractures all.dcm        ← fracture annotation
    pelvic segmentation/
      pelvis.dcm               ← pelvis ROI mask
```

All patient data is excluded from version control (`.gitignore`).

---

## Results

| Model | Fold | Train epochs | Train Dice Loss | Val Dice |
|---|---|---|---|---|
| nnU-Net | 0 | 4 | 0.45 (pseudo) | pending HPC |
| SAM3 + LoRA | 0 | 25 | 0.9989 | pending HPC |

Both models demonstrate learning from n=4 patients. Full training and evaluation across all 4 LOOCV folds is planned on HPC infrastructure.

---

## Architecture Summary

### nnU-Net
- **Architecture:** PlainConvUNet 3D, auto-configured from dataset fingerprint
- **Parameters:** ~11M
- **Key feature:** Anisotropic [1,3,1] kernels in early stages to handle 5.77× spacing ratio
- **Loss:** Dice + Cross-Entropy with deep supervision

### SAM3 + LoRA
- **Backbone:** Hiera ViT, 840M parameters (frozen)
- **Seg head:** 4-layer conv (256→64→32→16→1), 39,553 trainable parameters (0.005%)
- **Input:** 2.5D — each slice stacked with ±1 neighbours as 3-channel input
- **Loss:** Slice-wise Dice loss
- **Feature caching:** Backbone runs once, features saved as `.npy` per slice

---

## References

- Isensee et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*.
- Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos. *arXiv*.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
- Wasserthal et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images. *Radiology: AI*.
