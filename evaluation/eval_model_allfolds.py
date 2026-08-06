import os, sys, json, csv, argparse
import numpy as np, nibabel as nib
from scipy import ndimage

sys.path.insert(0, "${PROJECT_ROOT}")
from eval_complete import dice_3d, prec_rec, surf_and_hd
from surface_rp import surface_rp

ROOT = "${PROJECT_ROOT}"
GT_DIR = f"{ROOT}/nnsam_workdir/raw/Dataset002_FracturePelvis/labelsTr"
SPLITS = f"{ROOT}/test_splits_3way.json"

IOU_LEVELS = [0.1, 0.2, 0.5]          # reported alongside "any overlap"
MIN_LESION = 100


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def lesion_metrics_multi(p, g, iou_levels, min_vox, gt_min=None):
    """Lesion-level detection at SEVERAL matching criteria at once.

    Predictions are filtered at `min_vox`; ground-truth detection TARGETS are
    filtered at `gt_min` (defaults to `min_vox`). Setting gt_min > min_vox
    restricts the analysis to larger fractures (severity stratification): in
    that mode only lesion RECALL is interpretable, because a prediction that
    correctly hits a now-excluded small fracture is counted as a false positive,
    artificially depressing precision and F1.

    Returns dict with, for 'any' and each IoU level:
        les_rec_<k>, les_prec_<k>, les_f1_<k>, plus n_gt, n_pred.
    """
    if gt_min is None:
        gt_min = min_vox
    st = np.ones((3, 3, 3), int)
    gl, ng = ndimage.label(g, st)
    pl, npr = ndimage.label(p, st)
    g_sizes = np.bincount(gl.ravel())
    p_sizes = np.bincount(pl.ravel())
    gids = set(np.where(g_sizes >= gt_min)[0]) - {0}      # GT targets: gt_min
    pids = set(np.where(p_sizes >= min_vox)[0]) - {0}     # predictions: min_vox

    det = {"any": set(), **{f"iou{t}": set() for t in iou_levels}}
    mpr = {"any": set(), **{f"iou{t}": set() for t in iou_levels}}

    for gi in gids:
        gm = (gl == gi)
        gsz = g_sizes[gi]
        for pi in [int(x) for x in np.unique(pl[gm]) if x > 0 and int(x) in pids]:
            inter = int((gm & (pl == pi)).sum())
            if inter == 0:
                continue
            det["any"].add(gi); mpr["any"].add(pi)
            union = gsz + p_sizes[pi] - inter
            iou = inter / union if union > 0 else 0.0
            for t in iou_levels:
                if iou >= t:
                    det[f"iou{t}"].add(gi); mpr[f"iou{t}"].add(pi)

    ng_e, np_e = len(gids), len(pids)
    out = {"n_gt": ng_e, "n_pred": np_e}
    for k in det:
        r = len(det[k]) / ng_e if ng_e else float("nan")
        pr = len(mpr[k]) / np_e if np_e else float("nan")
        out[f"les_rec_{k}"] = r
        out[f"les_prec_{k}"] = pr
        out[f"les_f1_{k}"] = f1(pr, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="prediction dir suffix, e.g. nnunet | samed | swinunetr_v2_gated | medsam3")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--min-lesion", type=int, default=MIN_LESION)
    ap.add_argument("--gt-min-lesion", type=int, default=None,
                    help="restrict detection TARGETS to GT lesions >= this size "
                         "(severity stratification). Affects lesion recall/precision/F1 "
                         "only; voxel and surface metrics are unchanged. Under this mode "
                         "only RECALL is interpretable -- precision and F1 are depressed "
                         "because correct hits on excluded small fractures count as FP.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt_min = a.gt_min_lesion if a.gt_min_lesion is not None else a.min_lesion

    splits = json.load(open(SPLITS))
    rows = []
    for fold in a.folds:
        pred_dir = f"{ROOT}/test_eval/fold{fold}/pred_{a.model}"
        if not os.path.isdir(pred_dir):
            print(f"!! fold {fold}: no dir {pred_dir}", flush=True); continue
        test_ids = splits[fold]["test"]
        print(f"=== [{a.model}] fold {fold}: {len(test_ids)} test patients ===", flush=True)
        for pid in test_ids:
            pp, gp = f"{pred_dir}/{pid}.nii.gz", f"{GT_DIR}/{pid}.nii.gz"
            if not (os.path.exists(pp) and os.path.exists(gp)):
                print(f"  SKIP {pid} (missing)"); continue
            pn = nib.load(pp)
            p = np.asarray(pn.dataobj) > 0
            g = np.asarray(nib.load(gp).dataobj) > 0
            sp = pn.header.get_zooms()[:3]

            if p.sum() == 0:
                print(f"  !! {pid}: EMPTY PREDICTION (0 foreground voxels)", flush=True)

            # voxel + surface metrics are ALWAYS on the full mask (not gt_min-filtered)
            d = dice_3d(p, g); pr, rc = prec_rec(p, g)
            sd, hd = surf_and_hd(p, g, sp); srec, sprec = surface_rp(p, g, sp)
            lm = lesion_metrics_multi(p, g, IOU_LEVELS, a.min_lesion, gt_min)

            row = dict(model=a.model, fold=fold, pid=pid, dice=d, prec=pr, rec=rc,
                       sdice1=sd[1.0], sdice2=sd[2.0], sdice5=sd[5.0],
                       srec1=srec[1.0], srec2=srec[2.0], srec5=srec[5.0],
                       sprec1=sprec[1.0], sprec2=sprec[2.0], sprec5=sprec[5.0],
                       hd95=hd, **lm)
            rows.append(row)
            print(f"  {pid}: Dice={d:.3f}  lesRec any={lm['les_rec_any']:.2f} "
                  f"@.1={lm['les_rec_iou0.1']:.2f} @.2={lm['les_rec_iou0.2']:.2f} "
                  f"@.5={lm['les_rec_iou0.5']:.2f}  n_pred={lm['n_pred']}", flush=True)

    if not rows:
        print("NO PREDICTIONS FOUND"); return

    tag = f"_gtmin{gt_min}" if a.gt_min_lesion is not None else ""
    out_csv = a.out or f"{ROOT}/test_eval/metrics_{a.model}{tag}_allfolds.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def mu(k):
        v = [r[k] for r in rows if isinstance(r[k], float) and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    def sd_(k):
        v = [r[k] for r in rows if isinstance(r[k], float) and r[k] == r[k]]
        if not v: return 0.0
        m_ = sum(v) / len(v)
        return (sum((x - m_) ** 2 for x in v) / len(v)) ** 0.5

    print("\n" + "=" * 72)
    print(f"  {a.model.upper()}  —  {len(rows)} patients, folds {sorted(set(r['fold'] for r in rows))}")
    if a.gt_min_lesion is not None:
        print(f"  SEVERITY MODE: GT targets >= {gt_min} vox  (report RECALL only)")
    print("=" * 72)
    print(f"{'metric':<28}{'mean':>10}{'std':>10}")
    print("-" * 72)
    for label, key in [
        ("Voxel Dice", "dice"),
        ("Voxel Precision", "prec"),
        ("Voxel Recall", "rec"),
        ("Surface Dice@1mm", "sdice1"), ("Surface Dice@2mm", "sdice2"), ("Surface Dice@5mm", "sdice5"),
        ("Surface Recall@1mm", "srec1"), ("Surface Recall@2mm", "srec2"), ("Surface Recall@5mm", "srec5"),
        ("Surface Prec@1mm", "sprec1"), ("Surface Prec@2mm", "sprec2"), ("Surface Prec@5mm", "sprec5"),
        ("HD95 (mm)", "hd95"),
    ]:
        print(f"{label:<28}{mu(key):>10.4f}{sd_(key):>10.4f}")

    print("-" * 72)
    print("LESION-LEVEL DETECTION — sensitivity to the matching criterion:")
    if a.gt_min_lesion is not None:
        print("  (severity mode: RECALL interpretable; precision/F1 depressed by design)")
    print(f"{'criterion':<18}{'recall':>10}{'precision':>12}{'F1':>10}")
    for k, nice in [("any", "any overlap"), ("iou0.1", "IoU >= 0.1"),
                    ("iou0.2", "IoU >= 0.2"), ("iou0.5", "IoU >= 0.5")]:
        print(f"{nice:<18}{mu(f'les_rec_{k}'):>10.4f}{mu(f'les_prec_{k}'):>12.4f}"
              f"{mu(f'les_f1_{k}'):>10.4f}")
    print("-" * 72)
    print(f"{'GT lesions/patient':<28}{mu('n_gt'):>10.2f}")
    print(f"{'Predicted lesions/patient':<28}{mu('n_pred'):>10.2f}")
    n_empty = sum(1 for r in rows if r["n_pred"] == 0)
    if n_empty:
        print(f"\n  !! {n_empty}/{len(rows)} patients had EMPTY predictions")
    print("=" * 72)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
