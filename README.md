# Pelvic Fracture Segmentation on Photon-Counting CT

Code accompanying the master's thesis *Automatic Pelvic Fracture Segmentation on
Photon-Counting CT* (KU Leuven, Faculty of Engineering Science, 2026).

This work benchmarks four segmentation models — nnU-Net, SwinUNETR, SAM 1, and
SAM 3 — on pelvic fracture-line segmentation in photon-counting CT (PCCT), and
proposes a surface-based instance-matching criterion for evaluating thin
anatomical structures.

---

## Overview

Pelvic fracture lines are thin, elongated, and extremely sparse: annotated
fracture voxels occupy a median of 0.008 % of a scan. This makes them a poor fit
for conventional overlap-based segmentation metrics, and a hard target for
models designed around compact structures.

Three modelling paradigms are compared on 68 expert-annotated PCCT
examinations under a common cross-validation protocol:

| Model | Input | Initialisation | Trained on target data |
|---|---|---|---|
| nnU-Net | 3D patches | random | full network (126.3 M) |
| SwinUNETR | 3D patches | self-supervised medical-CT pre-training | full network (62.2 M) |
| SAM 1 (ViT-B) | 2D axial slices | SA-1B natural images | LoRA + mask decoder (3.93 M) |
| SAM 3 | 2D axial slices | large-scale concept segmentation | LoRA + mask decoder (4.74 M) |

Both SAM models are adapted prompt-free via low-rank adaptation, following
[SAMed](https://github.com/hitachinsk/SAMed).

---

## Data availability

**The imaging data and annotations cannot be shared.** They are patient scans
acquired at UZ Leuven and are governed by the study's ethics approval. Model
predictions and pelvic bone masks are derived from those scans and are likewise
not distributed.

The repository contains code only. Reproducing the results requires an
equivalent annotated PCCT dataset in nnU-Net raw format:

    raw/DatasetXXX_Name/
        imagesTr/Case_0000.nii.gz
        labelsTr/Case.nii.gz
        dataset.json

---

## Repository layout

    preprocessing/    dataset construction, pelvic masking, 2D slice generation
    training/         per-model training scripts
    inference/        prediction scripts
    evaluation/       boundary and instance metrics, incl. the surface criterion
    visualisation/    figure generation
    slurm/            HPC job scripts (SLURM)
    checkpoints/      SAM 1 and SAM 3 LoRA weights (see below)

---

## Installation

    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt

nnU-Net and MONAI are installed as dependencies. The SAM backbones are obtained
from their upstream repositories; see `checkpoints/README.md`.

---

## Pipeline

### 1. Preprocessing

Restrict volumes to the pelvic bone mask (dilated by four voxels, exterior set
to −1000 HU):

    python preprocessing/build_masked_dataset.py

Generate 2D axial slices for the SAM-family models, with foreground
oversampling at the slice level:

    python preprocessing/gen_samed_fold.py --fold 0

### 2. Training

    # nnU-Net (custom trainer with early stopping on validation loss)
    nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_ES

    # SwinUNETR
    python training/train_swinunetr_fold.py --fold 0

    # SAM 1 / SAM 3
    python training/train_sam3.py --fold 0 --rank 4 --base_lr 1e-3 --dice_param 0.8

### 3. Inference

    python inference/predict_sam3.py --fold 0 --ckpt <path> --out <dir>

nnU-Net inference uses a sliding-window stride of 1.0; predictions on this task
proved sensitive to this setting (see the thesis appendix).

### 4. Evaluation

    # boundary metrics + IoU-based instance metrics
    python evaluation/eval_model_allfolds.py --model <name> --min-lesion 300

    # surface-based instance metrics
    python evaluation/eval_lesion_nsd.py --model <name> --tau 2 5

---

## Surface-based instance matching

Overlap-based instance matching fails on thin structures: for a fixed relative
boundary error, IoU falls far faster for an elongated object than a compact one,
because a greater proportion of its volume lies near its surface.

`evaluation/eval_lesion_nsd.py` implements an alternative that extends the
tolerance principle of the Normalised Surface Dice to individual instances. For
a reference instance *g*, the surface recall

    r_τ(g) = |{ s ∈ ∂g : d(s, ∂P) ≤ τ }| / |∂g|

is the fraction of its boundary lying within τ millimetres of the predicted
surface, with surface precision defined symmetrically. An instance counts as
detected when r_τ(g) ≥ θ. Two parameters therefore control the criterion: τ
(how close the surfaces must be, in mm) and θ (how much of the instance must be
localised).

---

## Model checkpoints

LoRA adapters and mask decoders for the SAM-family models are included under
`checkpoints/`, one per cross-validation fold. The frozen backbones are not
redistributed and must be obtained from the upstream repositories.

nnU-Net and SwinUNETR checkpoints exceed GitHub's file-size limit and are
available on request.

---

## Citation

    @mastersthesis{bogucki2026pcct,
      title  = {Automatic Pelvic Fracture Segmentation on Photon-Counting CT},
      author = {Bogucki, Bartosz},
      school = {KU Leuven, Faculty of Engineering Science},
      year   = {2026}
    }

---

## Acknowledgements

Supervised by Prof. Maarten De Vos, with daily guidance from Konstantinos
Kontras and Tim Hermans. Clinical annotations by Stijn De Bondt (UZ Leuven).

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet),
[MONAI](https://github.com/Project-MONAI/MONAI),
[SAM](https://github.com/facebookresearch/segment-anything), and
[SAMed](https://github.com/hitachinsk/SAMed).
