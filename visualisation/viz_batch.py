#!/usr/bin/env python3
"""Batch-run viz_fractures.py over all test patients for one or more models.

Usage:
  python viz_batch.py --models sam3_final samed_50 nnunet swinunetr_v2_gated \
                      --modes gallery overview --folds 0 1 2 3 4
  python viz_batch.py --models sam3_final --modes gallery --no-gt      # predictions only
  python viz_batch.py --models sam3_final --patients Fracture_100 Fracture_017

Writes to viz_out_{model}/ (or viz_out_{model}_nogt/ with --no-gt).
"""
import argparse, json, os, subprocess, sys

R = "${PROJECT_ROOT}"
IMG = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"
LBL = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/labelsTr"
PY = f"{R}/envs/fracture/bin/python"
VIZ = f"{R}/viz_fractures.py"

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True,
                help="prediction dir names under test_eval/fold{F}/pred_<model>")
ap.add_argument("--modes", nargs="+", default=["gallery", "overview"],
                choices=["all", "metrics", "montage", "detail", "gallery", "overview", "3d"])
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--patients", nargs="+", default=None,
                help="restrict to specific patient IDs (default: all test patients)")
ap.add_argument("--no-gt", action="store_true",
                help="pass an empty GT so only predictions are drawn (visual comparison)")
ap.add_argument("--min-voxels", type=int, default=100)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

splits = json.load(open(f"{R}/test_splits_3way.json"))

# build (fold, pid) work list
work = []
for f in a.folds:
    for pid in splits[f]["test"]:
        if a.patients and pid not in a.patients:
            continue
        work.append((f, pid))

if not work:
    sys.exit("no patients matched")

print(f"{len(work)} patient(s) x {len(a.models)} model(s) x {len(a.modes)} mode(s)")

for model in a.models:
    suffix = "_nogt" if a.no_gt else ""
    out = f"{R}/viz_out_{model}{suffix}"
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {model} -> {out} ===")
    for fold, pid in work:
        pred = f"{R}/test_eval/fold{fold}/pred_{model}/{pid}.nii.gz"
        img = f"{IMG}/{pid}_0000.nii.gz"
        gt = f"{LBL}/{pid}.nii.gz"
        if not os.path.exists(pred):
            print(f"  skip {pid}: no prediction at {pred}")
            continue
        for mode in a.modes:
            cmd = [PY, VIZ, "--image", img, "--gt", gt, "--pred", pred,
                   "--out", out, "--pid", pid, "--mode", mode,
                   "--min-voxels", str(a.min_voxels)]
            if a.no_gt:
                # point --gt at an all-zero mask so nothing is drawn as GT/missed
                cmd += ["--gt", f"{R}/_empty_mask/{pid}.nii.gz"]
            print("  " + " ".join(cmd[2:]) if a.dry_run else f"  {pid} [{mode}]")
            if not a.dry_run:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"    FAILED: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
print("\ndone.")
