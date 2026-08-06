#!/usr/bin/env python3
"""Gate a prediction directory against per-patient pelvis masks (strict intersection).

Usage: gate_predictions.py --pred DIR --out DIR
  --pred : directory of Fracture_*.nii.gz predictions (original geometry)
  --out  : output directory for gated predictions (created)

Reports per-patient % of predicted voxels removed. Errors LOUDLY on shape mismatch
rather than silently copying (which would fake a 0% removal).
"""
import argparse, glob, os, sys
import numpy as np, nibabel as nib

R = "${PROJECT_ROOT}"
PELV = f"{R}/nnunet_eval/pelvis_masks/pelvis_masks_cropped"

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

preds = sorted(glob.glob(f"{a.pred}/Fracture_*.nii.gz"))
if not preds:
    sys.exit(f"ERROR: no predictions found in {a.pred}")

removed_fracs, mismatches, nomask = [], [], []
tot_before = tot_after = 0

for pp in preds:
    pid = os.path.basename(pp).replace(".nii.gz", "")
    mp = f"{PELV}/{pid}_pelvis.nii.gz"
    pn = nib.load(pp)
    pred = np.asarray(pn.dataobj) > 0
    before = int(pred.sum())

    if not os.path.exists(mp):
        nib.save(pn, f"{a.out}/{pid}.nii.gz")
        nomask.append(pid)
        print(f"{pid}: NO MASK -> copied unchanged ({before} vox)")
        continue

    pelv = np.asarray(nib.load(mp).dataobj) > 0
    if pred.shape != pelv.shape:
        # DO NOT silently copy — that fakes 0% removed. Flag it.
        mismatches.append((pid, pred.shape, pelv.shape))
        print(f"{pid}: !! SHAPE MISMATCH pred{pred.shape} vs pelv{pelv.shape} -> SKIPPED")
        continue

    gated = pred & pelv
    after = int(gated.sum())
    nib.save(nib.Nifti1Image(gated.astype(np.uint8), pn.affine, pn.header),
             f"{a.out}/{pid}.nii.gz")
    frac = 100.0 * (before - after) / max(before, 1)
    removed_fracs.append(frac)
    tot_before += before; tot_after += after
    print(f"{pid}: {before:>8d} -> {after:>8d}  ({frac:5.1f}% removed)")

print("\n" + "="*60)
print(f"gated {len(removed_fracs)} patients, {len(nomask)} no-mask, {len(mismatches)} MISMATCH")
if removed_fracs:
    print(f"per-patient % removed : mean {np.mean(removed_fracs):.1f}  "
          f"median {np.median(removed_fracs):.1f}  max {np.max(removed_fracs):.1f}")
    print(f"pooled % removed      : {100.0*(tot_before-tot_after)/max(tot_before,1):.1f}")
if mismatches:
    print("!! MISMATCHES (not gated — investigate):")
    for pid,ps,ms in mismatches: print(f"   {pid}: pred{ps} pelv{ms}")
print(f"-> {a.out}")
