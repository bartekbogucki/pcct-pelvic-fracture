#!/usr/bin/env python3
"""Gallery for ONE slice: a zoomed panel per predicted component on that slice.

Each row is one predicted component found on the requested slice:
  left  = clean windowed crop around that component (no annotation)
  right = same crop with the prediction outlined (and GT, with --show-gt)

This is the per-lesion zoom of viz_all's gallery mode, but pinned to a chosen
slice and driven by the PREDICTION rather than the ground truth.

Usage:
  python viz_slice_gallery.py --pid Fracture_091 --slice 1457 \
        --pred test_eval/fold4/pred_nnunet_s10_gated
  # orientation controls if the view does not match your other figures:
  #   --no-transpose      do not transpose the slice
  #   --origin upper      use image-style origin
  #   --flipud / --fliplr
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

PR_RGB = (1.00, 0.30, 0.20)
GT_RGB = (0.15, 0.80, 0.25)

ap = argparse.ArgumentParser()
ap.add_argument("--pid", required=True)
ap.add_argument("--pred", required=True)
ap.add_argument("--slice", type=int, required=True)
ap.add_argument("--min-voxels", type=int, default=300)
ap.add_argument("--min-slice-px", type=int, default=20,
                help="ignore components with fewer than this many pixels ON the slice")
ap.add_argument("--max-panels", type=int, default=6)
ap.add_argument("--pad", type=int, default=45)
ap.add_argument("--lo", type=float, default=-245.); ap.add_argument("--hi", type=float, default=1484.)
ap.add_argument("--show-gt", action="store_true")
ap.add_argument("--fill", action="store_true")
# orientation controls, to match other figures
ap.add_argument("--no-transpose", action="store_true")
ap.add_argument("--origin", default="lower", choices=["lower", "upper"])
ap.add_argument("--flipud", action="store_true")
ap.add_argument("--fliplr", action="store_true")
ap.add_argument("--out", default=f"{R}/viz_out_slice_gallery")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def chunked_bool(path):
    n = nib.load(path)
    v = np.zeros(n.shape, bool)
    for z0 in range(0, n.shape[2], 128):
        z1 = min(z0 + 128, n.shape[2])
        v[:, :, z0:z1] = np.asanyarray(n.dataobj[:, :, z0:z1]) > 0
    return v


def orient(x):
    if not a.no_transpose:
        x = x.T
    if a.flipud:
        x = np.flipud(x)
    if a.fliplr:
        x = np.fliplr(x)
    return x


z = a.slice
c_raw = np.asanyarray(nib.load(f"{IMG}/{a.pid}_0000.nii.gz").dataobj[:, :, z]).astype(np.float32)
pred = chunked_bool(f"{R}/{a.pred}/{a.pid}.nii.gz")
gt = chunked_bool(f"{LBL}/{a.pid}.nii.gz")

# 3D component filter (matches evaluation), then label so panels follow components
lab, n = ndimage.label(pred)
if n:
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= a.min_voxels)[0]; keep = keep[keep > 0]
    print(f"component filter (>= {a.min_voxels} vox): {n} -> {len(keep)} comps")
    lab = np.where(np.isin(lab, keep), lab, 0)

lab2 = orient(lab[:, :, z])
g2 = orient(gt[:, :, z])
c2 = np.clip(orient(c_raw), a.lo, a.hi); c2 = (c2 - a.lo) / (a.hi - a.lo)

ids, counts = np.unique(lab2[lab2 > 0], return_counts=True)
sel = [(int(i), int(c)) for i, c in zip(ids, counts) if c >= a.min_slice_px]
sel.sort(key=lambda t: -t[1])
sel = sel[:a.max_panels]
print(f"slice z={z}: {len(ids)} predicted components present, "
      f"{len(sel)} with >= {a.min_slice_px} px (showing {len(sel)})")

if not sel:
    print("nothing to draw on this slice"); raise SystemExit(0)

nrow = len(sel)
fig, axes = plt.subplots(nrow, 2, figsize=(11, 5.2 * nrow), squeeze=False)

for r, (cid, npx) in enumerate(sel):
    m = (lab2 == cid)
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min()-a.pad), min(c2.shape[0], ys.max()+a.pad)
    x0, x1 = max(0, xs.min()-a.pad), min(c2.shape[1], xs.max()+a.pad)
    sub, msub = c2[y0:y1, x0:x1], m[y0:y1, x0:x1]
    gsub = g2[y0:y1, x0:x1]

    axL, axR = axes[r][0], axes[r][1]
    axL.imshow(sub, cmap="gray", origin=a.origin, vmin=0, vmax=1)
    axL.set_title(f"{a.pid}  z={z}  (component {r+1})", fontsize=11)
    axL.axis("off")

    axR.imshow(sub, cmap="gray", origin=a.origin, vmin=0, vmax=1)
    if a.fill:
        ov = np.zeros((*msub.shape, 4), np.float32); ov[msub] = (*PR_RGB, 0.55)
        axR.imshow(ov, origin=a.origin)
    else:
        axR.contour(msub, levels=[.5], colors=[PR_RGB], linewidths=1.8)
    if a.show_gt and gsub.any():
        axR.contour(gsub, levels=[.5], colors=[GT_RGB], linewidths=1.8)
    axR.set_title(f"prediction ({npx} px on slice)", fontsize=11)
    axR.axis("off")

h = [Line2D([0], [0], color=PR_RGB, lw=2, label="prediction")]
if a.show_gt: h.append(Line2D([0], [0], color=GT_RGB, lw=2, label="ground truth"))
axes[0][1].legend(handles=h, loc="lower right", fontsize=9, framealpha=.85)

model = a.pred.rstrip("/").split("/")[-1].replace("pred_", "")
fig.suptitle(f"{a.pid} — {model}, axial slice z={z}", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.98])
p = f"{a.out}/{a.pid}_z{z}_{model}_gallery.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("wrote", p)
