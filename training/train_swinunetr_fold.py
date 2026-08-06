#!/usr/bin/env python3
"""
train_swinunetr_vsc.py — SwinUNETR (MONAI) SSL-pretrained, VSC version.
Matches nnU-Net protocol: select on LOWEST VAL LOSS, early stopping patience=100.
Uses 3-way splits fold 0: train (fit) + val (select). TEST is withheld (never used here).
Runs on Dataset002_FracturePelvis (pelvis-masked).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import monai
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped, Activationsd, AsDiscreted)

# ── Paths (VSC) ──
ROOT = Path("${PROJECT_ROOT}")
DATA = ROOT/"nnsam_workdir/raw/Dataset002_FracturePelvis"   # masked data
IMG_DIR = DATA/"imagesTr"; LBL_DIR = DATA/"labelsTr"
SPLITS = ROOT/"splits_final_3way.json"
OUT_DIR = None  # set after FOLD is parsed (fold-specific)
SSL_PATH = ROOT/"swinunetr_out"/"ssl_pretrained_weights.pth"

# ── Config ──
TARGET_SPACING=(0.5,0.5,0.5); A_MIN,A_MAX=-245.0,1484.0
ROI=(96,96,96); FEATURE_SIZE=48
MAX_EPOCHS=1000          # high cap; early stopping will end it
ES_PATIENCE=100         # match nnU-Net ES
POS_RATIO=8; NUM_SAMPLES=4
BATCH_SIZE=1; SW_BATCH=2; LR=1e-4; WEIGHT_DECAY=1e-5; NUM_WORKERS=4
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--fold', type=int, required=True)
FOLD = _ap.parse_args().fold
OUT_DIR = ROOT/f"swinunetr_folds/fold{FOLD}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_data_list(ids):
    items=[]
    for pid in ids:
        ct=IMG_DIR/f"{pid}_0000.nii.gz"; seg=LBL_DIR/f"{pid}.nii.gz"
        if ct.exists() and seg.exists():
            items.append({"image":str(ct),"label":str(seg),"pid":pid})
        else:
            print(f"  MISSING {pid}: ct={ct.exists()} seg={seg.exists()}")
    return items

train_tf=Compose([
    LoadImaged(keys=["image","label"]), EnsureChannelFirstd(keys=["image","label"]),
    Orientationd(keys=["image","label"],axcodes="RAS"),
    Spacingd(keys=["image","label"],pixdim=TARGET_SPACING,mode=("bilinear","nearest")),
    ScaleIntensityRanged(keys=["image"],a_min=A_MIN,a_max=A_MAX,b_min=0.0,b_max=1.0,clip=True),
    CropForegroundd(keys=["image","label"],source_key="image"),
    RandCropByPosNegLabeld(keys=["image","label"],label_key="label",spatial_size=ROI,
        pos=POS_RATIO,neg=1,num_samples=NUM_SAMPLES,image_key="image",image_threshold=0),
    RandFlipd(keys=["image","label"],spatial_axis=[0],prob=0.2),
    RandFlipd(keys=["image","label"],spatial_axis=[1],prob=0.2),
    RandFlipd(keys=["image","label"],spatial_axis=[2],prob=0.2),
    RandRotate90d(keys=["image","label"],prob=0.2,max_k=3),
    RandShiftIntensityd(keys=["image"],offsets=0.10,prob=0.5),
    EnsureTyped(keys=["image","label"]),
])
val_tf=Compose([
    LoadImaged(keys=["image","label"]), EnsureChannelFirstd(keys=["image","label"]),
    Orientationd(keys=["image","label"],axcodes="RAS"),
    Spacingd(keys=["image","label"],pixdim=TARGET_SPACING,mode=("bilinear","nearest")),
    ScaleIntensityRanged(keys=["image"],a_min=A_MIN,a_max=A_MAX,b_min=0.0,b_max=1.0,clip=True),
    CropForegroundd(keys=["image","label"],source_key="image"),
    EnsureTyped(keys=["image","label"]),
])
post_pred=Compose([Activationsd(keys="pred",softmax=True),AsDiscreted(keys="pred",argmax=True,to_onehot=2)])
post_label=Compose([AsDiscreted(keys="label",to_onehot=2)])

def load_ssl_weights(model, ssl_path):
    import re
    ckpt=torch.load(ssl_path,map_location="cpu",weights_only=False)
    state=ckpt.get("model",ckpt.get("state_dict",ckpt))
    model_keys=set(model.swinViT.state_dict().keys()); mapped={}
    for k,v in state.items():
        nk=k
        for pref in ("module.","encoder.","swinViT.","swin_vit."):
            if nk.startswith(pref): nk=nk[len(pref):]
        nk=re.sub(r"(layers\d+)\.(\d+)\.(\d+)\.blocks",r"\1.\2.blocks",nk)
        nk=re.sub(r"(layers\d+)\.(\d+)\.(\d+)\.downsample",r"\1.\2.downsample",nk)
        if nk in model_keys: mapped[nk]=v
    missing,unexpected=model.swinViT.load_state_dict(mapped,strict=False)
    print(f"  SSL load: {len(mapped)} matched, {len(missing)} missing, {len(unexpected)} unexpected")
    return len(mapped)>0

def main():
    print("="*64); print(f"  SwinUNETR | Dataset002 (masked) | fold {FOLD}"); print("="*64)
    print(f"  device {device} MONAI {monai.__version__}")
    splits=json.load(open(SPLITS))
    train_ids=splits[FOLD]["train"]; val_ids=splits[FOLD]["val"]
    # strip "Fracture_" prefix? No — files are Fracture_0XX_0000.nii.gz, ids ARE Fracture_0XX
    print(f"  train {len(train_ids)}  val {len(val_ids)}  (TEST withheld, not loaded)")
    train_files=build_data_list(train_ids); val_files=build_data_list(val_ids)

    train_loader=DataLoader(Dataset(train_files,train_tf),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
    val_loader=DataLoader(Dataset(val_files,val_tf),batch_size=1,shuffle=False,num_workers=2,pin_memory=True)

    model=SwinUNETR(in_channels=1,out_channels=2,feature_size=FEATURE_SIZE,use_checkpoint=True,spatial_dims=3).to(device)
    ok=load_ssl_weights(model,SSL_PATH) if SSL_PATH.exists() else False
    print(f"  init: {'SSL-pretrained' if ok else 'random'}")

    # ── FREEZE swinViT encoder (SSL-pretrained backbone), train full decoder ──
    for p in model.swinViT.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  FROZEN backbone: {n_total-n_train:,} frozen / {n_train:,} trainable ({100*n_train/n_total:.1f}% trainable)")

    loss_fn=DiceCELoss(to_onehot_y=True,softmax=True,squared_pred=True,batch=True,smooth_nr=1e-5,smooth_dr=1e-5)
    optimizer=torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=MAX_EPOCHS)
    dice_metric=DiceMetric(include_background=False,reduction="mean")
    scaler=torch.cuda.amp.GradScaler()

    # SELECTION on lowest val loss + early stopping (match nnU-Net)
    best_val_loss=np.inf; best_val_dice=-1.0; epochs_since_improve=0
    history=[]
    for epoch in range(MAX_EPOCHS):
        model.train(); epoch_loss=0.0; nstep=0
        for batch in train_loader:
            x=batch["image"].to(device); y=batch["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss=loss_fn(model(x),y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_loss+=loss.item(); nstep+=1
        scheduler.step(); epoch_loss/=max(nstep,1)

        # validation every epoch
        model.eval(); dice_metric.reset(); vloss_sum=0.0; vn=0
        with torch.no_grad():
            for batch in val_loader:
                x=batch["image"].to(device); y=batch["label"].to(device)
                with torch.cuda.amp.autocast():
                    logits=sliding_window_inference(x,ROI,SW_BATCH,model,overlap=0.5)
                    vloss=loss_fn(logits,y)
                vloss_sum+=vloss.item(); vn+=1
                out=[post_pred({"pred":p})["pred"] for p in decollate_batch(logits)]
                lab=[post_label({"label":l})["label"] for l in decollate_batch(y)]
                dice_metric(y_pred=out,y=lab)
        vdice=dice_metric.aggregate().item(); vloss_mean=vloss_sum/max(vn,1)

        improved = vloss_mean < best_val_loss - 1e-6
        tag=""
        if improved:
            best_val_loss=vloss_mean; best_val_dice=vdice; epochs_since_improve=0
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),
                        "val_loss":vloss_mean,"val_dice":vdice}, OUT_DIR/"best_valloss.pth")
            tag=" *BEST(val_loss)*"
        else:
            epochs_since_improve+=1
        print(f"  [ep {epoch:3d}] tr_loss {epoch_loss:.4f} | val_loss {vloss_mean:.4f} val_dice {vdice:.4f}"
              f" | since_improve {epochs_since_improve}{tag}", flush=True)
        history.append({"epoch":epoch,"train_loss":epoch_loss,"val_loss":vloss_mean,"val_dice":vdice})
        json.dump(history, open(OUT_DIR/"history.json","w"), indent=2)

        if epochs_since_improve>=ES_PATIENCE:
            print(f"  [ES] val_loss no improve for {ES_PATIENCE} epochs (best {best_val_loss:.4f}). Stop at {epoch}.", flush=True)
            break

    print(f"\nDONE. Best val_loss={best_val_loss:.4f} (val_dice={best_val_dice:.4f}) -> {OUT_DIR}/best_valloss.pth")

if __name__=="__main__":
    main()
