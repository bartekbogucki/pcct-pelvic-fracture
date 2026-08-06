#!/usr/bin/env python3
"""Lesion-level, surface-based (event) metrics at one or more distance tolerances.

MOTIVATION
----------
Standard lesion-level detection matches a predicted component to a ground-truth
lesion when their IoU exceeds a threshold. For thin, elongated structures such as
fracture lines, IoU collapses under even small boundary errors, so the criterion is
unfairly strict. This script instead matches on SURFACE PROXIMITY: a boundary
element counts as agreeing if the other structure's boundary lies within tau mm.

DEFINITIONS (tolerance tau, in mm)
----------------------------------
For each GT lesion g:
    surf_recall(g) = |{s in dG_g : d(s, dP) <= tau}| / |dG_g|
For each predicted component p:
    surf_prec(p)   = |{s in dP_p : d(s, dG) <= tau}| / |dP_p|

Aggregation, both reported:
  * CONTINUOUS  : mean surf_recall over lesions; mean surf_prec over predictions;
                  lesion NSD = 2*R*P/(R+P) from those means.
  * EVENT-BASED : lesion detected if surf_recall(g) >= theta; prediction a TP if
                  surf_prec(p) >= theta. Reported at 'any' (fraction > 0) and at
                  theta in THETAS. The sweep is the surface analogue of an IoU sweep.

Surface point-to-surface distances are computed ONCE per patient with a KD-tree and
re-thresholded for each tolerance, so `--tau 1 2 5` costs the same as `--tau 2`.

USAGE
-----
  python eval_lesion_nsd.py --model sam3_final_gated
  python eval_lesion_nsd.py --model sam3_final_gated --tau 1 2 5
  python eval_lesion_nsd.py --model sam3_final_gated --only-any
  python eval_lesion_nsd.py --model sam3_final_gated --folds 0

Writes test_eval/lesion_nsd_<model>_tau<t>.csv per tolerance and prints a summary.
"""
import argparse, csv, json, os, sys
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

R = "${PROJECT_ROOT}"
LBL = f"{R}/nnsam_workdir/raw/Dataset002_FracturePelvis/labelsTr"

THETAS = [0.1, 0.25, 0.5]      # boundary-fraction thresholds for the event criterion


def surface_points(mask, spacing):
    """Boundary voxels of `mask` as physical (mm) coordinates."""
    if not mask.any():
        return np.empty((0, 3), np.float32)
    er = ndimage.binary_erosion(mask, border_value=0)
    idx = np.argwhere(mask & ~er)
    return idx.astype(np.float32) * np.asarray(spacing, np.float32)


def lesion_surface_distances(gt, pred, spacing, min_voxels, gt_min):
    """Per-lesion and per-prediction surface distance arrays (mm).

    Returns (rec_d, prc_d): lists of 1-D arrays. rec_d[i] holds, for GT lesion i,
    the distance from each of its surface points to the nearest predicted surface
    point; prc_d[j] is the analogous array for predicted component j. Thresholding
    these at any tau gives the surface recall / precision fractions.
    """
    gt_lab, n_gt = ndimage.label(gt)
    pr_lab, n_pr = ndimage.label(pred)

    gt_ids, pr_ids = [], []
    if n_gt:
        gs = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
        gt_ids = [i for i in range(1, n_gt + 1) if gs[i] >= gt_min]
    if n_pr:
        ps = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)
        pr_ids = [i for i in range(1, n_pr + 1) if ps[i] >= min_voxels]

    gt_keep = np.isin(gt_lab, gt_ids) if gt_ids else np.zeros_like(gt, bool)
    pr_keep = np.isin(pr_lab, pr_ids) if pr_ids else np.zeros_like(pred, bool)

    gt_pts = surface_points(gt_keep, spacing)
    pr_pts = surface_points(pr_keep, spacing)
    tree_gt = cKDTree(gt_pts) if len(gt_pts) else None
    tree_pr = cKDTree(pr_pts) if len(pr_pts) else None

    rec_d = []
    for i in gt_ids:
        pts = surface_points(gt_lab == i, spacing)
        if len(pts) == 0:
            continue
        if tree_pr is None:
            rec_d.append(np.full(len(pts), np.inf, np.float32))
        else:
            d, _ = tree_pr.query(pts, workers=-1)
            rec_d.append(d.astype(np.float32))

    prc_d = []
    for j in pr_ids:
        pts = surface_points(pr_lab == j, spacing)
        if len(pts) == 0:
            continue
        if tree_gt is None:
            prc_d.append(np.full(len(pts), np.inf, np.float32))
        else:
            d, _ = tree_gt.query(pts, workers=-1)
            prc_d.append(d.astype(np.float32))

    return rec_d, prc_d


def fractions_at(dists, tau):
    """Fraction of each component's surface lying within tau."""
    return np.array([float(np.mean(d <= tau)) for d in dists], np.float64)


def f1(r, p):
    return (2 * r * p / (r + p)) if (r + p) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="prediction dir name under test_eval/fold{F}/pred_<model>")
    ap.add_argument("--tau", nargs="+", type=float, default=[2.0],
                    help="tolerance(s) in mm; several cost no extra time")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--min-voxels", type=int, default=300,
                    help="ignore components smaller than this (default 300, below the "
                         "5th percentile of ground-truth fracture size)")
    ap.add_argument("--gt-min-voxels", type=int, default=None,
                    help="restrict detection TARGETS to GT components >= this size "
                         "(severity stratification; independent of --min-voxels, which "
                         "filters predictions). Default: same as --min-voxels. "
                         "NB: precision is not interpretable under GT filtering; report recall only.")
    ap.add_argument("--only-any", action="store_true",
                    help="report only the any-surface-contact criterion, skipping the "
                         "theta sweep (note: this does not reduce runtime)")
    a = ap.parse_args()
    gt_min = a.gt_min_voxels if a.gt_min_voxels is not None else a.min_voxels

    splits = json.load(open(f"{R}/test_splits_3way.json"))
    rows = {t: [] for t in a.tau}

    print(f"=== {a.model}  min_voxels={a.min_voxels}  tau={a.tau}  folds={a.folds} ===",
          flush=True)

    for fold in a.folds:
        for pid in splits[fold]["test"]:
            pp = f"{R}/test_eval/fold{fold}/pred_{a.model}/{pid}.nii.gz"
            gp = f"{LBL}/{pid}.nii.gz"
            if not os.path.exists(pp):
                print(f"  skip {pid}: no prediction"); continue
            gim = nib.load(gp)
            spacing = gim.header.get_zooms()[:3]
            gt = np.asanyarray(gim.dataobj) > 0
            pred = np.asanyarray(nib.load(pp).dataobj) > 0

            # distances computed ONCE, reused for every tolerance
            rec_d, prc_d = lesion_surface_distances(gt, pred, spacing, a.min_voxels, gt_min)

            for tau in a.tau:
                rec = fractions_at(rec_d, tau)
                prc = fractions_at(prc_d, tau)
                row = {"model": a.model, "tau": tau, "fold": fold, "pid": pid,
                       "n_gt": len(rec), "n_pred": len(prc),
                       "surf_rec_mean": float(rec.mean()) if len(rec) else float("nan"),
                       "surf_prec_mean": float(prc.mean()) if len(prc) else float("nan")}
                row["lesion_nsd"] = (f1(row["surf_rec_mean"], row["surf_prec_mean"])
                                     if len(rec) and len(prc) else float("nan"))
                # "any" = at least one boundary element within tau (strict > 0).
                # NB: `rec >= 0.0` would be trivially true for every lesion.
                row["rec_any"] = float(np.mean(rec > 0)) if len(rec) else float("nan")
                row["prec_any"] = float(np.mean(prc > 0)) if len(prc) else float("nan")
                row["f1_any"] = (f1(row["rec_any"], row["prec_any"])
                                 if len(rec) and len(prc) else float("nan"))
                if not a.only_any:
                    for th in THETAS:
                        r_ = float(np.mean(rec >= th)) if len(rec) else float("nan")
                        p_ = float(np.mean(prc >= th)) if len(prc) else float("nan")
                        row[f"rec_th{th}"] = r_
                        row[f"prec_th{th}"] = p_
                        row[f"f1_th{th}"] = (f1(r_, p_) if len(rec) and len(prc)
                                             else float("nan"))
                rows[tau].append(row)

            r0 = rows[a.tau[0]][-1]
            print(f"  {pid}: GT {r0['n_gt']:>3} pred {r0['n_pred']:>4} | "
                  f"tau{a.tau[0]:g} surfRec {r0['surf_rec_mean']:.3f} "
                  f"surfPrec {r0['surf_prec_mean']:.3f} NSD {r0['lesion_nsd']:.3f}",
                  flush=True)

    for tau in a.tau:
        rr = rows[tau]
        if not rr:
            print(f"no rows for tau={tau}"); continue
        suffix = f"_gtmin{gt_min}" if gt_min != a.min_voxels else ""
        out = f"{R}/test_eval/lesion_nsd_{a.model}_tau{tau:g}{suffix}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys()))
            w.writeheader(); w.writerows(rr)

        def agg(k):
            v = [r[k] for r in rr if k in r and not np.isnan(r[k])]
            return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

        print("\n" + "=" * 68)
        print(f"  {a.model.upper()}  —  {len(rr)} patients, tau = {tau} mm, "
              f"min_voxels = {a.min_voxels}")
        print("=" * 68)
        for k, lbl in [("n_pred", "Predicted components per scan"),
                       ("surf_rec_mean", "Lesion surface recall (continuous)"),
                       ("surf_prec_mean", "Lesion surface precision (continuous)"),
                       ("lesion_nsd", "Lesion NSD")]:
            m, s = agg(k); print(f"  {lbl:38s} {m:.4f} ± {s:.4f}")
        print("-" * 68)
        print("  EVENT-BASED, surface hit criterion (fraction of boundary within tau):")
        print(f"  {'theta':>8} {'recall':>10} {'precision':>12} {'F1':>10}")
        r_, _ = agg("rec_any"); p_, _ = agg("prec_any"); f_, _ = agg("f1_any")
        print(f"  {'any':>8} {r_:>10.4f} {p_:>12.4f} {f_:>10.4f}")
        if not a.only_any:
            for th in THETAS:
                r_, _ = agg(f"rec_th{th}"); p_, _ = agg(f"prec_th{th}"); f_, _ = agg(f"f1_th{th}")
                print(f"  {th:>8.2f} {r_:>10.4f} {p_:>12.4f} {f_:>10.4f}")
        print("=" * 68)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
