#!/usr/bin/env python3
"""Fracture segmentation visualisation.

Modes: metrics | detail | gallery | overview | all
Every image mode supports --side-by-side, which draws the clean (image-only)
panel next to the overlaid one, so a reader can judge whether the fracture is
visible in the image before seeing what was predicted.

Examples
--------
  python viz_fractures.py --image IMG --gt GT --pred PRED --pid Fracture_100 \
      --out viz_out_sam3 --mode all --side-by-side

  python viz_fractures.py ... --mode gallery --side-by-side --hu-lo -245 --hu-hi 1484
"""
import os, gc, argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

GREEN, RED, BLUE = "#2ca02c", "#d62728", "#1f78d8"

# --------------------------------- I/O ---------------------------------------

def load_mask(path):
    im = nib.load(path)
    return (np.asanyarray(im.dataobj) > 0).astype(np.uint8), im.header.get_zooms()[:3]

def load_ct(path, lo=-245., hi=1484.):
    """HU window is parameterised; defaults match the dataset-driven window
    (0.5 / 99.5 foreground percentiles) used for training."""
    im = nib.load(path)
    a = np.asanyarray(im.dataobj).astype(np.float32, copy=False)
    np.clip(a, lo, hi, out=a); a -= lo; a /= (hi - lo)
    return a

# ------------------------- component table & metrics --------------------------

def comp_table(mask, min_voxels):
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, []
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    objs = ndimage.find_objects(lab)
    idx = np.arange(1, n + 1)
    coms = ndimage.center_of_mass(mask, lab, idx)
    return lab, [{"id": int(c), "size": int(sizes[c - 1]),
                  "com": coms[c - 1], "bbox": objs[c - 1]}
                 for c in idx if sizes[c - 1] >= min_voxels]

def dice_bool(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2 * int(np.logical_and(a, b).sum()) / s) if s else float("nan")

def compute_metrics(gt, pred, min_voxels, roi_pad):
    gt_lab, gt_c = comp_table(gt, min_voxels)
    pr_lab, pr_c = comp_table(pred, min_voxels)
    pr_by_id = {p["id"]: p for p in pr_c}

    inter = int(np.count_nonzero(np.logical_and(gt, pred)))
    gsum = int(gt.sum()) + int(pred.sum())
    global_dice = (2 * inter / gsum) if gsum else float("nan")

    keep = np.zeros(int(pr_lab.max()) + 1, bool)
    for i in pr_by_id:
        keep[i] = True

    rows = []
    for g in gt_c:
        bb = g["bbox"]
        gsub = (gt_lab[bb] == g["id"])
        psub = pr_lab[bb]
        touch = [int(t) for t in np.unique(psub[gsub]) if t != 0 and keep[t]]
        overlap = int(np.count_nonzero(gsub & np.isin(psub, touch))) if touch else 0
        detected = overlap > 0

        if detected:
            st = [min([bb[i].start] + [pr_by_id[t]["bbox"][i].start for t in touch]) for i in range(3)]
            sp_ = [max([bb[i].stop] + [pr_by_id[t]["bbox"][i].stop for t in touch]) for i in range(3)]
            ub = tuple(slice(st[i], sp_[i]) for i in range(3))
            lesion_dice = dice_bool(np.isin(pr_lab[ub], touch), gt_lab[ub] == g["id"])
        else:
            lesion_dice = 0.0

        rb = tuple(slice(max(0, bb[i].start - roi_pad), min(gt.shape[i], bb[i].stop + roi_pad))
                   for i in range(3))
        roi_dice = dice_bool(keep[pr_lab[rb]], gt[rb] > 0)

        rows.append({"gt_id": g["id"], "size": g["size"], "detected": detected,
                     "overlap": overlap, "n_touch": len(touch),
                     "lesion_dice": lesion_dice, "roi_dice": roi_dice, "bbox": bb})

    det = [r for r in rows if r["detected"]]
    s = {"global_dice": global_dice, "n_gt": len(gt_c), "n_pred": len(pr_c),
         "n_detected": len(det),
         "lesion_dice_matched": float(np.mean([r["lesion_dice"] for r in det])) if det else float("nan"),
         "lesion_dice_all": float(np.mean([r["lesion_dice"] for r in rows])) if rows else float("nan"),
         "roi_dice_mean": float(np.mean([r["roi_dice"] for r in rows])) if rows else float("nan")}
    del gt_lab, pr_lab
    gc.collect()
    return s, rows

# ---------------------------- plotting helpers --------------------------------

def ov_rgba(g2, p2):
    tp = (g2 > 0) & (p2 > 0); fp = (p2 > 0) & (g2 == 0); fn = (g2 > 0) & (p2 == 0)
    ov = np.zeros((*g2.shape, 4), np.float32)
    ov[tp] = [.17, .63, .17, .9]
    ov[fp] = [.84, .15, .16, .8]
    ov[fn] = [.12, .47, .85, .9]
    return ov

def circles(ax, m2, color, ls="-", lw=2., min_px=4, scale=1.6, base=7):
    lab, n = ndimage.label(m2)
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    drawn = 0
    for c in range(1, n + 1):
        s = int(sizes[c - 1])
        if s < min_px:
            continue
        cy, cx = ndimage.center_of_mass(lab == c)
        ax.add_patch(Circle((cx, cy), max(base, np.sqrt(s) * scale),
                            fill=False, ec=color, lw=lw, ls=ls))
        drawn += 1
    return drawn

def leg(ax, items, loc="lower right"):
    h = [Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=9, label=l)
         if k == "s" else Line2D([0], [0], color=c, lw=2, ls=ls, label=l)
         for (l, c, k, ls) in items]
    ax.legend(handles=h, loc=loc, fontsize=8, framealpha=.75)

OVERLAY_LEGEND = [("TP", GREEN, "s", "-"), ("FP", RED, "s", "-"), ("missed", BLUE, "s", "-")]

def _frame(ax, color, lw=2):
    for sp_ in ax.spines.values():
        sp_.set_edgecolor(color); sp_.set_linewidth(lw)
    ax.set_xticks([]); ax.set_yticks([])

# --------------------------------- figures ------------------------------------

def fig_gallery(ct, gt, pred, pid, out, rows, ncol=4, pad=45, side_by_side=False):
    """Every GT fracture at ITS OWN centre slice, zoomed.
    With --side-by-side each lesion gets: clean crop | overlaid crop."""
    if not rows:
        print("  [gallery] no GT lesions"); return
    rs = sorted(rows, key=lambda r: -r["size"])
    n = len(rs)
    per = 2 if side_by_side else 1
    cols = ncol * per
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, cols, figsize=(4.0 * cols, 4.3 * nrow), squeeze=False)
    axes = axes.ravel()

    for i, r in enumerate(rs):
        bb = r["bbox"]
        z = (bb[2].start + bb[2].stop) // 2
        x0, x1 = max(0, bb[0].start - pad), min(ct.shape[0], bb[0].stop + pad)
        y0, y1 = max(0, bb[1].start - pad), min(ct.shape[1], bb[1].stop + pad)
        c2 = ct[x0:x1, y0:y1, z].T
        g2 = gt[x0:x1, y0:y1, z].T
        p2 = pred[x0:x1, y0:y1, z].T
        ok = r["detected"]
        col = "#1a7a1a" if ok else "#b30000"

        if side_by_side:
            axc = axes[i * 2]
            axc.imshow(c2, cmap="gray", origin="lower")
            axc.set_title(f"GT#{r['gt_id']}  z={z} — image only", fontsize=9)
            _frame(axc, col)
            ax = axes[i * 2 + 1]
        else:
            ax = axes[i]

        ax.imshow(c2, cmap="gray", origin="lower")
        ax.imshow(ov_rgba(g2, p2), origin="lower")
        ax.set_title(f"GT#{r['gt_id']}  z={z}  {r['size']} vox\n"
                     f"{'detected' if ok else 'MISSED'}  lesionDice {r['lesion_dice']:.2f}",
                     fontsize=9, color=col)
        _frame(ax, col)

    for j in range(n * per, len(axes)):
        axes[j].axis("off")
    leg(axes[1 if side_by_side else 0], OVERLAY_LEGEND)
    extra = " | left = image only, right = overlay" if side_by_side else ""
    fig.suptitle(f"Patient {pid} — every GT fracture at its own centre slice "
                 f"({n} lesions; green frame = detected, red = missed){extra}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .95])
    p = f"{out}/{pid}_gallery.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [gallery] {p}")


def fig_detail(ct, gt, pred, pid, out, rows, zoom_pad=35, max_zooms=4, side_by_side=False):
    """Single slice through the largest GT fracture: circles, overlay, and zooms.
    With --side-by-side a clean panel is added and each zoom becomes clean|overlay."""
    if rows:
        big = max(rows, key=lambda r: r["size"])
        z = (big["bbox"][2].start + big["bbox"][2].stop) // 2
    else:
        z = ct.shape[2] // 2
    c2, g2, p2 = ct[:, :, z].T, gt[:, :, z].T, pred[:, :, z].T
    ov = ov_rgba(g2, p2)

    lab_g, ng = ndimage.label(g2)
    order, objs = [], []
    if ng:
        sz = np.bincount(lab_g.ravel(), minlength=ng + 1)[1:]
        objs = ndimage.find_objects(lab_g)
        order = sorted(range(1, ng + 1), key=lambda c: -sz[c - 1])[:max_zooms]

    per = 2 if side_by_side else 1
    n_top = 3 if side_by_side else 2                       # clean | circles | overlay
    n_zoom_panels = max(1, len(order)) * per
    ncol = max(n_top, n_zoom_panels)

    fig = plt.figure(figsize=(4.6 * ncol, 10))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.5, 1.], hspace=.16)

    # ---- top row: split the full width into n_top equal segments
    seg = ncol / n_top
    bounds = [int(round(i * seg)) for i in range(n_top + 1)]
    bounds[-1] = ncol
    slot = 0
    if side_by_side:
        axc = fig.add_subplot(gs[0, bounds[0]:bounds[1]])
        axc.imshow(c2, cmap="gray", origin="lower")
        axc.set_title(f"Image only (z={z})", fontsize=11); axc.axis("off")
        slot = 1
    ax1 = fig.add_subplot(gs[0, bounds[slot]:bounds[slot + 1]])
    ax1.imshow(c2, cmap="gray", origin="lower")
    a_ = circles(ax1, g2, GREEN, "-", 2.2, scale=1.8, base=9)
    b_ = circles(ax1, p2, RED, "--", 1.3, scale=1.4, base=6)
    ax1.set_title(f"Circles (z={z}) — GT: {a_}, predicted: {b_}", fontsize=11); ax1.axis("off")
    leg(ax1, [("GT fracture", GREEN, "l", "-"), ("Predicted", RED, "l", "--")])
    ax2 = fig.add_subplot(gs[0, bounds[slot + 1]:bounds[slot + 2]])
    ax2.imshow(c2, cmap="gray", origin="lower")
    ax2.imshow(ov, origin="lower")
    ax2.set_title("Mask overlay — TP / FP / missed", fontsize=11); ax2.axis("off")
    leg(ax2, OVERLAY_LEGEND)

    # ---- zoom row
    if not order:
        ax = fig.add_subplot(gs[1, :]); ax.axis("off")
        ax.text(.5, .5, "no GT fracture on this slice", ha="center", va="center")
    else:
        for i, c in enumerate(order):
            r_, cc = objs[c - 1]
            R0, R1 = max(0, r_.start - zoom_pad), min(c2.shape[0], r_.stop + zoom_pad)
            C0, C1 = max(0, cc.start - zoom_pad), min(c2.shape[1], cc.stop + zoom_pad)
            if side_by_side:
                axa = fig.add_subplot(gs[1, i * 2])
                axa.imshow(c2[R0:R1, C0:C1], cmap="gray", origin="lower")
                axa.set_title(f"zoom {i+1} — image only", fontsize=9); axa.axis("off")
                axb = fig.add_subplot(gs[1, i * 2 + 1])
            else:
                axb = fig.add_subplot(gs[1, i])
            axb.imshow(c2[R0:R1, C0:C1], cmap="gray", origin="lower")
            axb.imshow(ov[R0:R1, C0:C1], origin="lower")
            axb.set_title(f"zoom {i+1} — overlay (z={z})", fontsize=9); axb.axis("off")

    fig.suptitle(f"Patient {pid} — single-slice detail (every panel from z={z})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .96])
    p = f"{out}/{pid}_detail.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [detail] {p}")


def fig_overview(ct, gt, pred, pid, out, s, pct=97, gamma=1.6, dim=0.75, side_by_side=False):
    """Whole-volume projections. High-percentile CT projection so overlays stay readable.
    With --side-by-side the clean projections form a top row and the overlays a bottom row."""
    names = ["Axial (through Z)", "Coronal (through Y)", "Sagittal (through X)"]
    proj_axes = [2, 1, 0]
    nrow = 2 if side_by_side else 1
    fig, axes = plt.subplots(nrow, 3, figsize=(17, 6 * nrow), squeeze=False)

    for j, (axis, name) in enumerate(zip(proj_axes, names)):
        c = np.percentile(ct, pct, axis=axis).T
        c = (c ** gamma) * dim
        g = (gt.max(axis=axis) > 0).T
        p = (pred.max(axis=axis) > 0).T
        ov = np.zeros((*g.shape, 4), np.float32)
        ov[g & p] = [.17, .90, .17, 1.0]
        ov[p & ~g] = [1.0, .20, .20, .85]
        ov[g & ~p] = [.20, .60, 1.0, 1.0]

        if side_by_side:
            axc = axes[0, j]
            axc.imshow(c, cmap="gray", origin="lower", aspect="auto", vmin=0, vmax=1)
            axc.set_title(f"{name} — image only", fontsize=11); axc.axis("off")
            ax = axes[1, j]
        else:
            ax = axes[0, j]
        ax.imshow(c, cmap="gray", origin="lower", aspect="auto", vmin=0, vmax=1)
        ax.imshow(ov, origin="lower", aspect="auto")
        ax.set_title(f"{name} — overlay" if side_by_side else name, fontsize=11)
        ax.axis("off")

    leg(axes[1 if side_by_side else 0, 0],
        [("GT ∩ pred", GREEN, "s", "-"), ("pred only (FP)", RED, "s", "-"),
         ("GT only (missed)", BLUE, "s", "-")])
    fig.suptitle(f"Patient {pid} — whole-volume projections   |   GT comps {s['n_gt']}   "
                 f"predicted comps {s['n_pred']}   global Dice {s['global_dice']:.3f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    p = f"{out}/{pid}_overview.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [overview] {p}")

# ----------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="viz_out")
    ap.add_argument("--pid", default="case")
    ap.add_argument("--mode", default="all",
                    choices=["all", "metrics", "detail", "gallery", "overview"])
    ap.add_argument("--side-by-side", action="store_true",
                    help="draw the clean image next to each overlaid panel")
    ap.add_argument("--hu-lo", type=float, default=-245.)
    ap.add_argument("--hu-hi", type=float, default=1484.)
    ap.add_argument("--min-voxels", type=int, default=300)
    ap.add_argument("--roi-pad", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print(f"loading {a.pid} ...")
    gt, sp = load_mask(a.gt)
    pred, _ = load_mask(a.pred)
    print(f"  masks: shape {gt.shape}  spacing {tuple(round(float(x), 3) for x in sp)}")

    if a.min_voxels > 1:
        lab, n = ndimage.label(pred)
        if n:
            sizes = np.bincount(lab.ravel())
            keep = np.where(sizes >= a.min_voxels)[0]
            keep = keep[keep > 0]
            before_vox = int(pred.sum())
            pred = np.isin(lab, keep).astype(np.uint8)
            print(f"  component filter (>= {a.min_voxels} vox): "
                  f"{n} -> {len(keep)} components, "
                  f"{before_vox} -> {int(pred.sum())} voxels")
            del lab, sizes
            gc.collect()

    s, rows = compute_metrics(gt, pred, a.min_voxels, a.roi_pad)
    print("\n---------------- metrics ----------------")
    print(f"  global Dice               : {s['global_dice']:.4f}")
    print(f"  GT components             : {s['n_gt']}")
    print(f"  predicted components      : {s['n_pred']}")
    print(f"  GT detected               : {s['n_detected']} / {s['n_gt']}")
    print(f"  per-lesion Dice (matched) : {s['lesion_dice_matched']:.4f}")
    print(f"  per-lesion Dice (all GT)  : {s['lesion_dice_all']:.4f}   (misses count as 0)")
    print(f"  ROI-local Dice (mean)     : {s['roi_dice_mean']:.4f}   (pad {a.roi_pad} vox)")
    print("-----------------------------------------")
    for r in rows:
        print(f"   GT#{r['gt_id']:>4} size {r['size']:>7} "
              f"{'DETECTED' if r['detected'] else 'MISSED  '} "
              f"overlap {r['overlap']:>6}  preds_touching {r['n_touch']:>3}  "
              f"lesionDice {r['lesion_dice']:.3f}  roiDice {r['roi_dice']:.3f}")
    print()
    if a.mode == "metrics":
        return

    print(f"loading CT (window [{a.hu_lo:.0f}, {a.hu_hi:.0f}] HU) ...")
    ct = load_ct(a.image, lo=a.hu_lo, hi=a.hu_hi)

    modes = ["detail", "gallery", "overview"] if a.mode == "all" else [a.mode]
    if "detail" in modes:
        fig_detail(ct, gt, pred, a.pid, a.out, rows, side_by_side=a.side_by_side)
    if "gallery" in modes:
        fig_gallery(ct, gt, pred, a.pid, a.out, rows, side_by_side=a.side_by_side)
    if "overview" in modes:
        fig_overview(ct, gt, pred, a.pid, a.out, s, side_by_side=a.side_by_side)
    print("done.")


if __name__ == "__main__":
    main()
