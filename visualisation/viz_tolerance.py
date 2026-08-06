#!/usr/bin/env python3
"""Illustrate surface agreement within a tolerance, on one fracture slice.

Elements per tolerance tau:
  - GREEN line        : complete ground-truth boundary
  - RED line          : complete predicted boundary
  - shaded GREEN band : region within tau of the GT boundary
  - shaded RED band   : region within tau of the predicted boundary
  - BLUE region       : overlap of the two bands (within tau of BOTH) -- the
                        area where the surfaces agree to within tau

Usage:
  python viz_tolerance.py --pid Fracture_100 --slice 283 --taus 1 2 5 \
        --pred test_eval/fold2/pred_nnunet_s10
"""
import argparse, os
import numpy as np, nibabel as nib
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = "${PROJECT_ROOT}"
IMG = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"
LBL = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/labelsTr"

GT_RGB  = (0.16, 0.65, 0.18)    # ground-truth boundary : green
PR_RGB  = (0.84, 0.15, 0.16)    # predicted boundary    : red
AGR_RGB = (0.12, 0.45, 0.95)    # band overlap (agreement) : blue

ap = argparse.ArgumentParser()
ap.add_argument("--pid", required=True)
ap.add_argument("--taus", nargs="+", type=float, default=[1.0, 2.0, 5.0])
ap.add_argument("--pred", default="test_eval/fold0/pred_sam3_final",
                help="prediction dir (relative to project root)")
ap.add_argument("--min-voxels", type=int, default=300)
ap.add_argument("--lo", type=float, default=-245.); ap.add_argument("--hi", type=float, default=1484.)
ap.add_argument("--slice", type=int, default=None)
ap.add_argument("--pad", type=int, default=60)
ap.add_argument("--out", default=f"{R}/viz_out_tolerance")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

im = nib.load(f"{IMG}/{a.pid}_0000.nii.gz")
ct = np.asanyarray(im.dataobj).astype(np.float32)
sp = im.header.get_zooms()[:3]
gt = np.asanyarray(nib.load(f"{LBL}/{a.pid}.nii.gz").dataobj) > 0

pred = np.asanyarray(nib.load(f"{R}/{a.pred}/{a.pid}.nii.gz").dataobj) > 0
if a.min_voxels > 1:
    lab, n = ndimage.label(pred)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = np.where(sizes >= a.min_voxels)[0]; keep = keep[keep > 0]
        pred = np.isin(lab, keep)

z = a.slice if a.slice is not None else int(np.argmax(gt.sum(axis=(0, 1))))
c2 = np.clip(ct[:, :, z].T, a.lo, a.hi); c2 = (c2 - a.lo) / (a.hi - a.lo)
g2 = gt[:, :, z].T
p2 = pred[:, :, z].T

sy, sx = float(sp[1]), float(sp[0])
dist_to_gt = ndimage.distance_transform_edt(~g2, sampling=(sy, sx)) if g2.any() \
             else np.full_like(c2, np.inf, np.float32)
dist_to_pred = ndimage.distance_transform_edt(~p2, sampling=(sy, sx)) if p2.any() \
               else np.full_like(c2, np.inf, np.float32)

mask_for_crop = g2 | p2
if mask_for_crop.any():
    ys, xs = np.where(mask_for_crop)
    y0, y1 = max(0, ys.min()-a.pad), min(c2.shape[0], ys.max()+a.pad)
    x0, x1 = max(0, xs.min()-a.pad), min(c2.shape[1], xs.max()+a.pad)
else:
    y0, y1, x0, x1 = 0, c2.shape[0], 0, c2.shape[1]

ntau = len(a.taus)
fig, axes = plt.subplots(1, ntau, figsize=(7.2*ntau, 7.4), squeeze=False)
axes = axes.ravel()

for ax, tau in zip(axes, a.taus):
    ax.imshow(c2[y0:y1, x0:x1], cmap="gray", origin="lower", vmin=0, vmax=1)

    d_gt = dist_to_gt[y0:y1, x0:x1]
    d_pr = dist_to_pred[y0:y1, x0:x1]
    gt_band = d_gt <= tau
    pr_band = d_pr <= tau
    agree = gt_band & pr_band                       # overlap of the two bands

    ov = np.zeros((y1-y0, x1-x0, 4), np.float32)
    ov[gt_band & ~agree] = (*GT_RGB, 0.22)          # GT-only band
    ov[pr_band & ~agree] = (*PR_RGB, 0.18)          # pred-only band
    ov[agree]            = (*AGR_RGB, 0.55)         # agreement (blue)
    ax.imshow(ov, origin="lower")

    if g2[y0:y1, x0:x1].any():
        ax.contour(g2[y0:y1, x0:x1], levels=[.5], colors=[GT_RGB], linewidths=2.4)
    if p2[y0:y1, x0:x1].any():
        ax.contour(p2[y0:y1, x0:x1], levels=[.5], colors=[PR_RGB], linewidths=1.8)

    ax.set_title(f"tolerance = {tau:.0f} mm", fontsize=13)
    ax.axis("off")

handles = [Line2D([0],[0], color=GT_RGB, lw=2.4, label="GT boundary"),
           Line2D([0],[0], color=PR_RGB, lw=1.8, label="prediction boundary"),
           Patch(facecolor=(*GT_RGB,0.5), label=r"within $\tau$ of GT"),
           Patch(facecolor=(*PR_RGB,0.5), label=r"within $\tau$ of prediction"),
           Patch(facecolor=(*AGR_RGB,0.75), label=r"agreement (within $\tau$ of both)")]
axes[0].legend(handles=handles, loc="lower right", fontsize=8, framealpha=.88)

fig.suptitle(f"{a.pid} — surface agreement within tolerance (illustrative axial slice z={z})",
             fontsize=14)
fig.tight_layout(rect=[0,0,1,.93])
p = f"{a.out}/{a.pid}_tolerance_z{z}.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print("wrote", p)
print(f"in-plane spacing ({sx:.3f}, {sy:.3f}) mm")
for tau in a.taus:
    print(f"  tolerance {tau:.0f} mm = {tau/sx:.1f} x {tau/sy:.1f} px")
