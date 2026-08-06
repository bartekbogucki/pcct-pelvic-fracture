#!/usr/bin/env python3
"""How much of the annotated fracture is lost to intensity windowing?

Reports, over all annotated fracture voxels in the cohort:
  - the HU distribution (percentiles)
  - the fraction clipped below / above each candidate window
and writes a histogram with the window bounds marked.

Usage:
  python window_analysis.py
  python window_analysis.py --windows -200 1000 -245 1484 -1000 1479
"""
import argparse, glob, os
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "${PROJECT_ROOT}"
IMG = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"
LBL = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/labelsTr"

ap = argparse.ArgumentParser()
ap.add_argument("--windows", nargs="+", type=float,
                default=[-200, 1000, -245, 1484, -1000, 1479],
                help="flat list of lo hi pairs")
ap.add_argument("--out", default=f"{R}/viz_out_window")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
wins = [(a.windows[i], a.windows[i+1]) for i in range(0, len(a.windows), 2)]

vals = []
n_scans = 0
for lp in sorted(glob.glob(f"{LBL}/*.nii.gz")):
    pid = os.path.basename(lp).replace(".nii.gz", "")
    ip = f"{IMG}/{pid}_0000.nii.gz"
    if not os.path.exists(ip):
        continue
    gn = nib.load(lp); im = nib.load(ip)
    # stream in z-chunks to stay light
    for z0 in range(0, gn.shape[2], 128):
        z1 = min(z0 + 128, gn.shape[2])
        g = np.asanyarray(gn.dataobj[:, :, z0:z1]) > 0
        if not g.any():
            continue
        c = np.asanyarray(im.dataobj[:, :, z0:z1]).astype(np.float32)
        vals.append(c[g])
    n_scans += 1
    print(f"  {pid}", flush=True)

v = np.concatenate(vals)
print(f"\n{len(v):,} annotated fracture voxels across {n_scans} scans\n")

print("HU distribution of fracture voxels:")
for q in [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]:
    print(f"   p{q:<5} {np.percentile(v, q):9.1f} HU")
print(f"   mean  {v.mean():9.1f}   min {v.min():.0f}   max {v.max():.0f}\n")

print(f"{'window':<22}{'clipped low':>13}{'clipped high':>14}{'total':>10}")
for lo, hi in wins:
    below = float((v < lo).mean()); above = float((v > hi).mean())
    print(f"[{lo:.0f}, {hi:.0f}]{'':<8}{100*below:>12.2f}%{100*above:>13.2f}%"
          f"{100*(below+above):>9.2f}%")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(v, bins=200, range=(-500, 2000), color="#4878a8", alpha=.85)
cols = ["#d62728", "#2ca02c", "#ff7f0e"]
for (lo, hi), c in zip(wins, cols):
    ax.axvline(lo, color=c, ls="--", lw=1.6, label=f"[{lo:.0f}, {hi:.0f}]")
    ax.axvline(hi, color=c, ls="--", lw=1.6)
ax.set_xlabel("HU"); ax.set_ylabel("annotated fracture voxels")
ax.set_title(f"Intensity distribution of annotated fracture voxels "
             f"({len(v):,} voxels, {n_scans} scans)")
ax.legend(fontsize=9)
fig.tight_layout()
p = f"{a.out}/fracture_hu_histogram.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print(f"\nwrote {p}")
