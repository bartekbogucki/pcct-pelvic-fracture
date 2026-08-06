import os, sys, json, argparse, numpy as np, torch, nibabel as nib
sys.path.insert(0,"${PROJECT_ROOT}/SAMed")
from scipy.ndimage import zoom as ndzoom
from einops import repeat
from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam
ROOT="${PROJECT_ROOT}"
IMG_DIR=f"{ROOT}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"
TEST_SPLITS=f"{ROOT}/test_splits_3way.json"
SAM_CKPT=f"{ROOT}/SAMed/checkpoints/sam_vit_b_01ec64.pth"
IMG_SIZE=512; A_MIN,A_MAX=-245.0,1484.0; TARGET_ISO=0.5; NUM_CLASSES=1; RANK=4
def norm(img):
    img=np.clip(img,A_MIN,A_MAX).astype(np.float32); return (img-A_MIN)/(A_MAX-A_MIN)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--ckpt",default=None); ap.add_argument("--out",default=None)
    ap.add_argument("--vit",default="vit_b"); ap.add_argument("--sam_ckpt",default=None)
    a=ap.parse_args()
    F=a.fold
    LORA=a.ckpt or f"{ROOT}/samed_allfolds_results/fold{F}/best_valloss.pth"
    print(f"loading LoRA: {LORA}",flush=True)
    OUT=a.out or f"{ROOT}/test_eval/fold{F}/pred_samed"; os.makedirs(OUT,exist_ok=True)
    SC=a.sam_ckpt or SAM_CKPT
    sam,_=sam_model_registry[a.vit](image_size=IMG_SIZE,num_classes=NUM_CLASSES,checkpoint=SC,pixel_mean=[0,0,0],pixel_std=[1,1,1])
    net=LoRA_Sam(sam,RANK).cuda(); net.load_lora_parameters(LORA); net.eval()
    mm=(NUM_CLASSES>1)
    print(f"fold {F}: loaded {LORA}")
    test_ids=json.load(open(TEST_SPLITS))[F]["test"]
    for pid in test_ids:
        ip=f"{IMG_DIR}/{pid}_0000.nii.gz"
        if not os.path.exists(ip): print(f"MISSING {pid}"); continue
        ref=nib.load(ip); rs=ref.shape; raff=ref.affine
        sp=ref.header.get_zooms()[:3]; img=ref.get_fdata().astype(np.float32)
        zf=[sp[i]/TARGET_ISO for i in range(3)]
        iso=norm(ndzoom(img,zf,order=1)); vol=iso.transpose(2,1,0); D,H,W=vol.shape
        pred=np.zeros((D,H,W),np.uint8)
        with torch.no_grad():
            for z in range(D):
                sl=vol[z]; x,y=sl.shape
                slr=ndzoom(sl,(IMG_SIZE/x,IMG_SIZE/y),order=3) if (x!=IMG_SIZE or y!=IMG_SIZE) else sl
                inp=torch.from_numpy(slr).unsqueeze(0).unsqueeze(0).float().cuda()
                inp=repeat(inp,'b c h w -> b (r c) h w',r=3)
                out=net(inp,mm,IMG_SIZE)
                m=torch.argmax(torch.softmax(out['masks'],dim=1),dim=1)[0].cpu().numpy().astype(np.uint8)
                if m.shape[0]!=x or m.shape[1]!=y: m=ndzoom(m,(x/m.shape[0],y/m.shape[1]),order=0)
                pred[z]=m
        pred_xyz=pred.transpose(2,1,0)
        zfb=[rs[i]/pred_xyz.shape[i] for i in range(3)]
        po=ndzoom(pred_xyz,zfb,order=0).astype(np.uint8)
        if po.shape!=tuple(rs):
            fx=np.zeros(rs,np.uint8); s=tuple(slice(0,min(po.shape[i],rs[i])) for i in range(3)); fx[s]=po[s]; po=fx
        nib.save(nib.Nifti1Image(po,raff),f"{OUT}/{pid}.nii.gz")
        print(f"{pid}: pred saved, fg voxels={po.sum()}",flush=True)
    print(f"DONE fold {F} predictions")
if __name__=="__main__":
    main()
