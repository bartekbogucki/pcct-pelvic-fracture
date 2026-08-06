import os, sys, json, argparse, numpy as np, nibabel as nib, h5py
from scipy.ndimage import zoom
ROOT="${PROJECT_ROOT}"
DATA=f"{ROOT}/nnsam_workdir/raw/Dataset002_FracturePelvis"
IMG_DIR=f"{DATA}/imagesTr"; LBL_DIR=f"{DATA}/labelsTr"
SPLITS=f"{ROOT}/splits_final_3way.json"
A_MIN,A_MAX=-245.0,1484.0; TARGET_ISO=0.5; BG_RATIO=0.5; MAX_SLICES=400
def norm(img):
    img=np.clip(img,A_MIN,A_MAX).astype(np.float32); return (img-A_MIN)/(A_MAX-A_MIN)
def resample(img,lbl,sp):
    zf=[s/TARGET_ISO for s in sp]
    return zoom(img,zf,order=1),(zoom(lbl,zf,order=0)>0.5).astype(np.uint8)
def gen_slices(ids, dst_npz, listfile, seed):
    rng=np.random.RandomState(seed); os.makedirs(dst_npz,exist_ok=True)
    names=[]; total=0
    for pid in ids:
        ip=f"{IMG_DIR}/{pid}_0000.nii.gz"; lp=f"{LBL_DIR}/{pid}.nii.gz"
        if not (os.path.exists(ip) and os.path.exists(lp)): print(f"MISSING {pid}"); continue
        ni=nib.load(ip); img=ni.get_fdata().astype(np.float32); lbl=nib.load(lp).get_fdata()
        sp=ni.header.get_zooms()[:3]; img,lbl=resample(img,lbl,sp); img=norm(img)
        img=img.transpose(2,1,0); lbl=lbl.transpose(2,1,0)
        frac=np.where(lbl.any(axis=(1,2)))[0]
        if len(frac)==0: continue
        bone=set(np.where((img>0.3).any(axis=(1,2)))[0]); bg=sorted(bone-set(frac.tolist()))
        nbg=min(int(len(frac)*BG_RATIO),len(bg))
        sel=rng.choice(bg,size=nbg,replace=False) if nbg>0 else []
        keep=sorted(set(frac.tolist())|set(int(x) for x in sel))
        if len(keep)>MAX_SLICES:
            keep=sorted(set(frac.tolist())|set(int(x) for x in sel[:max(0,MAX_SLICES-len(frac))]))
        num=pid.replace("Fracture_","")
        for d in keep:
            nm=f"case{num}_slice{d:04d}"; np.savez(f"{dst_npz}/{nm}.npz",image=img[d],label=lbl[d])
            names.append(nm); total+=1
        print(f"{pid}: {len(keep)} slices (total {total})",flush=True)
    open(listfile,"w").write("\n".join(names)+"\n")
    return total
def gen_test_vols(ids, dst_h5, listfile):
    os.makedirs(dst_h5,exist_ok=True); names=[]
    for pid in ids:
        ip=f"{IMG_DIR}/{pid}_0000.nii.gz"; lp=f"{LBL_DIR}/{pid}.nii.gz"
        if not (os.path.exists(ip) and os.path.exists(lp)): continue
        ni=nib.load(ip); img=ni.get_fdata().astype(np.float32); lbl=nib.load(lp).get_fdata()
        sp=ni.header.get_zooms()[:3]; img,lbl=resample(img,lbl,sp); img=norm(img)
        img=img.transpose(2,1,0); lbl=lbl.transpose(2,1,0)
        num=pid.replace("Fracture_","")
        with h5py.File(f"{dst_h5}/case{num}.npy.h5","w") as f:
            f["image"]=img; f["label"]=lbl
        names.append(f"case{num}")
    open(listfile,"w").write("\n".join(names)+"\n")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fold",type=int,required=True); a=ap.parse_args()
    F=a.fold; DST=f"{ROOT}/samed_data_w245_fold{F}"
    os.makedirs(f"{DST}/lists",exist_ok=True)
    sp=json.load(open(SPLITS))
    tr=sp[F]["train"]; va=sp[F]["val"]
    print(f"fold {F}: {len(tr)} train, {len(va)} val patients")
    nt=gen_slices(tr,f"{DST}/train_npz",f"{DST}/lists/train.txt",seed=42)
    nv=gen_slices(va,f"{DST}/val_npz",f"{DST}/lists/val.txt",seed=43)
    gen_test_vols(va,f"{DST}/test_vol_h5",f"{DST}/lists/val_vol.txt")
    print(f"DONE fold {F}: {nt} train, {nv} val slices -> {DST}")
if __name__=="__main__":
    main()
