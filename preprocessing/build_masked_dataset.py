import numpy as np, nibabel as nib, glob, os, json, shutil
from scipy.ndimage import binary_dilation

SRC_IMG='${PROJECT_ROOT}/nnsam_workdir/raw/Dataset001_Fracture/imagesTr'
SRC_LBL='${PROJECT_ROOT}/nnsam_workdir/raw/Dataset001_Fracture/labelsTr'
PELV='${PROJECT_ROOT}/nnunet_eval/pelvis_masks/pelvis_masks_cropped'
DST='${PROJECT_ROOT}/nnsam_workdir/raw/Dataset002_FracturePelvis'
DILATE_ITERS=4   # ~4 voxel dilation buffer
AIR=-1000.0
EXCLUDE={'Fracture_035'}  # no pelvis mask

os.makedirs(f'{DST}/imagesTr', exist_ok=True)
os.makedirs(f'{DST}/labelsTr', exist_ok=True)

struct=np.ones((3,3,3),bool)
done=0; skipped=0
for ip in sorted(glob.glob(f'{SRC_IMG}/Fracture_*_0000.nii.gz')):
    pid=os.path.basename(ip).replace('_0000.nii.gz','')
    if pid in EXCLUDE: print(f'{pid}: EXCLUDED'); skipped+=1; continue
    mp=f'{PELV}/{pid}_pelvis.nii.gz'; lp=f'{SRC_LBL}/{pid}.nii.gz'
    if not os.path.exists(mp): print(f'{pid}: no pelvis mask, skip'); skipped+=1; continue
    ct_nib=nib.load(ip); ct=np.asarray(ct_nib.dataobj).astype(np.float32)
    pelv=np.asarray(nib.load(mp).dataobj)>0
    if ct.shape!=pelv.shape: print(f'{pid}: SHAPE MISMATCH, skip'); skipped+=1; continue
    # dilate pelvis mask
    pelv_d=binary_dilation(pelv, structure=struct, iterations=DILATE_ITERS)
    # air-fill outside dilated pelvis
    masked=ct.copy(); masked[~pelv_d]=AIR
    nib.save(nib.Nifti1Image(masked, ct_nib.affine, ct_nib.header), f'{DST}/imagesTr/{pid}_0000.nii.gz')
    # copy label unchanged
    shutil.copy(lp, f'{DST}/labelsTr/{pid}.nii.gz')
    inside=int(pelv.sum()); buffered=int(pelv_d.sum())
    print(f'{pid}: pelvis {inside} -> dilated {buffered} vox, CT air-masked outside')
    done+=1

# dataset.json
dj={"channel_names":{"0":"CT"},"labels":{"background":0,"fracture":1},
    "numTraining":done,"file_ending":".nii.gz"}
json.dump(dj, open(f'{DST}/dataset.json','w'), indent=2)
print(f'\nDONE: {done} masked, {skipped} skipped -> {DST}')
