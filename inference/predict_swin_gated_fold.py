import os, json, numpy as np, torch, nibabel as nib
import monai
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, EnsureTyped)
from scipy.ndimage import zoom as ndzoom

ROOT="${PROJECT_ROOT}"
import sys; FOLD=int(sys.argv[1])
IMG_DIR=f"{ROOT}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"
PELV_DIR=f"{ROOT}/nnunet_eval/pelvis_masks/pelvis_masks_cropped"
SPLITS=f"{ROOT}/test_splits_3way.json"
CKPT=f"{ROOT}/swinunetr_folds/fold{FOLD}/best_valloss.pth"
OUT=f"{ROOT}/test_eval/fold{FOLD}/pred_swinunetr_v2_gated"
TARGET_SPACING=(0.5,0.5,0.5); A_MIN,A_MAX=-200.0,1000.0; ROI=(96,96,96); FEATURE_SIZE=48; SW_BATCH=2
os.makedirs(OUT,exist_ok=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_ids=json.load(open(SPLITS))[FOLD]["test"]
print("test patients:", test_ids)
model=SwinUNETR(in_channels=1,out_channels=2,feature_size=FEATURE_SIZE,use_checkpoint=True,spatial_dims=3).to(device)
ck=torch.load(CKPT,map_location=device)
model.load_state_dict(ck["state_dict"]); model.eval()
print(f"loaded {CKPT} (epoch {ck.get('epoch','?')}, val_dice {ck.get('val_dice','?')})")
pre=Compose([
    LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"],axcodes="RAS"),
    Spacingd(keys=["image"],pixdim=TARGET_SPACING,mode="bilinear"),
    ScaleIntensityRanged(keys=["image"],a_min=A_MIN,a_max=A_MAX,b_min=0.0,b_max=1.0,clip=True),
    EnsureTyped(keys=["image"]),
])
for pid in test_ids:
    ip=f"{IMG_DIR}/{pid}_0000.nii.gz"
    if not os.path.exists(ip): print(f"MISSING {pid}"); continue
    ref=nib.load(ip); ref_shape=ref.shape; ref_affine=ref.affine
    d=pre({"image":ip})
    img=d["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits=sliding_window_inference(img,ROI,SW_BATCH,model,overlap=0.5)
        pred=torch.argmax(torch.softmax(logits,dim=1),dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    zf=[ref_shape[i]/pred.shape[i] for i in range(3)]
    pred_orig=ndzoom(pred,zf,order=0).astype(np.uint8)
    if pred_orig.shape!=tuple(ref_shape):
        fixed=np.zeros(ref_shape,np.uint8)
        sl=tuple(slice(0,min(pred_orig.shape[i],ref_shape[i])) for i in range(3))
        fixed[sl]=pred_orig[sl]; pred_orig=fixed
    # ---- anatomical gating: keep predictions only inside the pelvis mask ----
    raw_vox = int(pred_orig.sum())
    mp = f"{PELV_DIR}/{pid}_pelvis.nii.gz"
    if os.path.exists(mp):
        pelv = np.asarray(nib.load(mp).dataobj) > 0
        if pelv.shape == pred_orig.shape:
            pred_orig = (pred_orig * pelv).astype(np.uint8)
            gated_vox = int(pred_orig.sum())
            drop = 100*(1 - gated_vox/max(raw_vox,1))
            print(f"{pid}: pred {raw_vox} -> {gated_vox} vox after pelvis gating "
                  f"({drop:.1f}% removed)", flush=True)
        else:
            print(f"{pid}: SHAPE MISMATCH pred {pred_orig.shape} vs pelvis {pelv.shape} "
                  f"-> NOT gated", flush=True)
    else:
        print(f"{pid}: NO pelvis mask -> ungated ({raw_vox} vox)", flush=True)
    nib.save(nib.Nifti1Image(pred_orig,ref_affine),f"{OUT}/{pid}.nii.gz")
print("DONE swinunetr fold0 test predictions")
