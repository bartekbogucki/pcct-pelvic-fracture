#!/usr/bin/env python3
"""Side-by-side: left = clean windowed slice (no annotation),
                 right = same slice with a model's prediction overlaid.

Usage:
  python viz_pair.py --pid Fracture_091 --slice 1457 \
        --pred test_eval/fold2/pred_sam3_w245_gated
"""
import argparse, os
import numpy as np, nibabel as nib
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "${PROJECT_ROOT}"
IMG = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"
LBL = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/labelsTr"

PR_RGB = (1.00, 0.30, 0.20)   # prediction : red
GT_RGB = (0.15, 0.80, 0.25)   # optional GT : green

ap = argparse.ArgumentParser()
ap.add_argument("--pid", required=True)
ap.add_argument("--pred", required=True, help="prediction dir (relative to project root)")
ap.add_argument("--slice", type=int, required=True)
ap.add_argument("--lo", type=float, default=-245.); ap.add_argument("--hi", type=float, default=1484.)
ap.add_argument("--min-voxels", type=int, default=300)
ap.add_argument("--show-gt", action="store_true", help="also outline ground truth (green) on the right")
ap.add_argument("--fill", action="store_true", help="fill prediction instead of outline")
ap.add_argument("--pad", type=int, default=60,
                help="crop margin (px) around the fracture; use --pad 0 for the full slice")
ap.add_argument("--out", default=f"{R}/viz_out_pair")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# load ONLY the requested slice from the image (memory-light: a few MB, not ~2 GB)
_img = nib.load(f"{IMG}/{a.pid}_0000.nii.gz")
c_raw = np.asanyarray(_img.dataobj[:, :, a.slice]).astype(np.float32)
# read the prediction in z-chunks, keeping only a boolean volume in memory
_pn = nib.load(f"{R}/{a.pred}/{a.pid}.nii.gz")
_shape = _pn.shape
pred = np.zeros(_shape, dtype=bool)
_CH = 128
for _z0 in range(0, _shape[2], _CH):
    _z1 = min(_z0 + _CH, _shape[2])
    pred[:, :, _z0:_z1] = np.asanyarray(_pn.dataobj[:, :, _z0:_z1]) > 0
_gn = nib.load(f"{LBL}/{a.pid}.nii.gz")
gt = np.zeros(_gn.shape, dtype=bool)
for _z0 in range(0, _gn.shape[2], 128):
    _z1 = min(_z0 + 128, _gn.shape[2])
    gt[:, :, _z0:_z1] = np.asanyarray(_gn.dataobj[:, :, _z0:_z1]) > 0

# component filter to match evaluation
if a.min_voxels > 1:
    lab, n = ndimage.label(pred)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = np.where(sizes >= a.min_voxels)[0]; keep = keep[keep > 0]
        before = int(pred.sum())
        pred = np.isin(lab, keep)
        print(f"component filter (>= {a.min_voxels} vox): {n} -> {len(keep)} comps, "
              f"{before} -> {int(pred.sum())} voxels")

z = a.slice
c2 = np.clip(c_raw.T, a.lo, a.hi); c2 = (c2 - a.lo) / (a.hi - a.lo)
p2 = pred[:, :, z].T
g2 = gt[:, :, z].T if gt is not None else None

# zoom crop: follow the PREDICTION; fall back to GT if the prediction has
# nothing on this slice. --pad 0 shows the full slice.
if a.pad > 0:
    focus = p2 if p2.any() else (g2 if g2 is not None else p2)
    if focus.any():
        ys, xs = np.where(focus)
        y0, y1 = max(0, ys.min()-a.pad), min(c2.shape[0], ys.max()+a.pad)
        x0, x1 = max(0, xs.min()-a.pad), min(c2.shape[1], xs.max()+a.pad)
        c2, p2 = c2[y0:y1, x0:x1], p2[y0:y1, x0:x1]
        if g2 is not None: g2 = g2[y0:y1, x0:x1]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7.6))

axL.imshow(c2, cmap="gray", origin="lower", vmin=0, vmax=1)
axL.set_title(f"{a.pid} — axial slice z={z}", fontsize=12)
axL.axis("off")

axR.imshow(c2, cmap="gray", origin="lower", vmin=0, vmax=1)
if a.fill and p2.any():
    ov = np.zeros((*p2.shape, 4), np.float32); ov[p2] = (*PR_RGB, 0.55)
    axR.imshow(ov, origin="lower")
elif p2.any():
    axR.contour(p2, levels=[.5], colors=[PR_RGB], linewidths=1.8)
if a.show_gt and g2 is not None and g2.any():
    axR.contour(g2, levels=[.5], colors=[GT_RGB], linewidths=1.8)
model_name = a.pred.rstrip("/").split("/")[-1].replace("pred_", "")
axR.set_title(f"prediction: {model_name}", fontsize=12)
axR.axis("off")

handles = [Line2D([0],[0], color=PR_RGB, lw=2, label="prediction")]
if a.show_gt: handles.append(Line2D([0],[0], color=GT_RGB, lw=2, label="ground truth"))
axR.legend(handles=handles, loc="lower right", fontsize=9, framealpha=.85)

fig.tight_layout()
p = f"{a.out}/{a.pid}_z{z}_{model_name}.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print("wrote", p)
print(f"slice z={z}: predicted fg px on slice = {int(p2.sum())}")
