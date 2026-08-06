#!/usr/bin/env python3
"""Show the preprocessing chain on one slice:
   left   = raw reconstruction (Dataset001, no masking, full HU range)
   middle = raw slice under the bone window
   right  = pelvis-masked slice under the bone window (Dataset002, what models see)

Usage:
  python viz_preproc.py --pid Fracture_100
  python viz_preproc.py --pid Fracture_100 --slice 391 --contour
"""
import argparse, os
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "${PROJECT_ROOT}"
RAW = f"{R}/nnsam_workdir/raw/Dataset001_Fracture/imagesTr"          # unmasked
MSK = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"    # pelvis-masked
LBL = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/labelsTr"
PELV = f"{R}/nnunet_eval/pelvis_masks/pelvis_masks_cropped"

ap = argparse.ArgumentParser()
ap.add_argument("--pid", required=True)
ap.add_argument("--lo", type=float, default=-245.)
ap.add_argument("--hi", type=float, default=1484.)
ap.add_argument("--slice", type=int, default=None,
                help="axial slice index (default: slice with most fracture voxels)")
ap.add_argument("--contour", action="store_true", help="outline the fracture in red")
ap.add_argument("--outline-pelvis", action="store_true",
                help="outline the pelvic mask on the raw panels")
ap.add_argument("--out", default=f"{R}/viz_out_preproc")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

raw = np.asanyarray(nib.load(f"{RAW}/{a.pid}_0000.nii.gz").dataobj).astype(np.float32)
msk = np.asanyarray(nib.load(f"{MSK}/{a.pid}_0000.nii.gz").dataobj).astype(np.float32)
gt = np.asanyarray(nib.load(f"{LBL}/{a.pid}.nii.gz").dataobj) > 0
pelv = None
if a.outline_pelvis and os.path.exists(f"{PELV}/{a.pid}_pelvis.nii.gz"):
    pelv = np.asanyarray(nib.load(f"{PELV}/{a.pid}_pelvis.nii.gz").dataobj) > 0

z = a.slice if a.slice is not None else int(np.argmax(gt.sum(axis=(0, 1))))
sr, sm = raw[:, :, z].T, msk[:, :, z].T
gm = gt[:, :, z].T
pm = pelv[:, :, z].T if pelv is not None else None


def win(x, lo, hi):
    y = np.clip(x, lo, hi)
    return (y - lo) / (hi - lo)

panels = [
    ("Raw reconstruction", sr, None),
    ("Intensity clipping", sr, (a.lo, a.hi)),
    ("Pelvis-masked", sm, (a.lo, a.hi)),
]

fig, axes = plt.subplots(1, 3, figsize=(19, 6.8))
for ax, (title, img, w) in zip(axes, panels):
    if w is None:
        ax.imshow(img, cmap="gray", origin="lower",
                  vmin=float(img.min()), vmax=float(img.max()))
        sub = f"range [{img.min():.0f}, {img.max():.0f}] HU"
    else:
        ax.imshow(win(img, *w), cmap="gray", origin="lower", vmin=0, vmax=1)
        sub = f"{100.0*((img < w[0]) | (img > w[1])).mean():.1f}% of voxels clipped"
    if a.contour and gm.any():
        ax.contour(gm, levels=[.5], colors="#397c33", linewidths=1.3)
    if pm is not None and w is None:
        ax.contour(pm, levels=[.5], colors="#911fd8", linewidths=1.0, linestyles="--")
    ax.set_title(title, fontsize=16, pad=18)

    ax.text(
        0.5, 1.01, sub,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13
    )
    ax.axis("off")

fig.tight_layout()
p = f"{a.out}/{a.pid}_preproc_z{z}.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("wrote", p)
print(f"raw   : min {raw.min():.0f}  max {raw.max():.0f}  "
      f"air-fraction {float((raw <= -999).mean()):.3f}")
print(f"masked: min {msk.min():.0f}  max {msk.max():.0f}  "
      f"air-fraction {float((msk <= -999).mean()):.3f}")
