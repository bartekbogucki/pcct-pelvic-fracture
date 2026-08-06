#!/usr/bin/env python3
"""Training/validation loss curves for the final five-fold runs.

One panel per model, one line per fold, with the selected (minimum validation
loss) epoch marked. Reads history.json for SwinUNETR and the SAM models; nnU-Net
writes its own progress.png per fold, parsed from its training log if present.

Usage:
  python plot_curves.py
  python plot_curves.py --sam3-tag w245 --sam1-dir samed_allfolds_50
"""
import argparse, json, os, re, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "02_training_logs"

ap = argparse.ArgumentParser()
ap.add_argument("--sam3-tag", default="w245", help="sam3_lora_results/fold{F}_{tag}")
ap.add_argument("--sam1-dir", default="samed_allfolds_50")
ap.add_argument("--swin-dir", default="swinunetr_folds")
ap.add_argument("--out", default=f"{R}/viz_out_curves")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

NN_LOG = (f"{R}/nnsam_workdir/results/Dataset002_FracturePelvis/"
          "nnUNetTrainer_ES__nnUNetPlans__3d_fullres/fold_{f}/training_log_*.txt")


def from_history(path):
    if not os.path.exists(path):
        return None
    h = json.load(open(path))
    ep = [r["epoch"] for r in h]
    tr = [r.get("train_loss") for r in h]
    va = [r.get("val_loss") for r in h]
    return ep, tr, va


def from_nnunet_log(fold):
    g = sorted(glob.glob(NN_LOG.format(f=fold)))
    if not g:
        return None
    txt = open(g[-1], errors="ignore").read()
    tr = [float(x) for x in re.findall(r"train_loss\s+(-?[\d.]+)", txt)]
    va = [float(x) for x in re.findall(r"val_loss\s+(-?[\d.]+)", txt)]
    n = min(len(tr), len(va))
    if n == 0:
        return None
    # nnU-Net optimises CE + (-Dice); the loss defined in the methods uses
    # (1 - Dice). The two differ only by an additive constant and give identical
    # gradients, so we shift by +1 to plot all models under the same convention.
    tr = [t + 1.0 for t in tr[:n]]
    va = [v + 1.0 for v in va[:n]]
    return list(range(n)), tr, va


models = [
    ("nnU-Net",   lambda f: from_nnunet_log(f)),
    ("SwinUNETR", lambda f: from_history(f"{R}/{a.swin_dir}/fold{f}/history.json")),
    ("SAM 1",     lambda f: from_history(
        f"{R}/{a.sam1_dir}/fold{f}/Synapse_512_pretrain_vit_b_epo50_bs8_lr0.005/history.json")),
    ("SAM 3",     lambda f: from_history(f"{R}/sam3_lora_results/fold{f}_{a.sam3_tag}/history.json")),
]

fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
cols = plt.cm.viridis(np.linspace(0, .85, 5))

for ax, (name, getter) in zip(axes, models):
    got = 0
    for f in range(5):
        d = getter(f)
        if d is None:
            continue
        ep, tr, va = d
        got += 1
        ax.plot(ep, tr, color=cols[f], lw=1.4, alpha=.9, label=f"fold {f}")
        ax.plot(ep, va, color=cols[f], lw=1.4, alpha=.9, ls="--")
        vv = [v for v in va if v is not None]
        if vv:
            bi = int(np.argmin([np.inf if v is None else v for v in va]))
            ax.plot(ep[bi], va[bi], "o", color=cols[f], ms=6, mec="k", mew=.7)
    ax.set_title(f"{name}" + ("" if got else "  (no history found)"), fontsize=12)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.grid(alpha=.3)
    if got:
        ax.legend(fontsize=8)
    print(f"{name}: {got}/5 folds found")

fig.suptitle("Training (solid) and validation (dashed) loss for the final five-fold runs; "
             "markers indicate the selected checkpoint", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .93])
p = f"{a.out}/training_curves.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
print("wrote", p)
