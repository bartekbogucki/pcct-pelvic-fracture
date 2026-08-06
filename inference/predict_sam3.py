import os, sys, json, argparse, numpy as np, torch, nibabel as nib
sys.path.insert(0, "${PROJECT_ROOT}")
sys.path.insert(0, "${PROJECT_ROOT}/SAMed")
sys.path.insert(0, "${PROJECT_ROOT}/MedSAM3")
from scipy.ndimage import zoom as ndzoom
from einops import repeat
from sam3_lora_seg import build_sam3_lora_seg

ROOT = "${PROJECT_ROOT}"
IMG_DIR = f"{ROOT}/nnsam_workdir/raw/Dataset002_FracturePelvis/imagesTr"
TEST_SPLITS = f"{ROOT}/test_splits_3way.json"
IMG_SIZE = 512; A_MIN, A_MAX = -200.0, 1000.0; TARGET_ISO = 0.5; NUM_CLASSES = 1

def norm(img):
    img = np.clip(img, A_MIN, A_MAX).astype(np.float32)
    return (img - A_MIN) / (A_MAX - A_MIN)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    F = a.fold
    os.makedirs(a.out, exist_ok=True)

    print(f"building SAM3 (rank {a.rank}) + loading {a.ckpt}", flush=True)
    net = build_sam3_lora_seg(rank=a.rank).cuda()
    net.load_lora_parameters(a.ckpt)
    net.eval()

    test_ids = json.load(open(TEST_SPLITS))[F]["test"]
    if a.limit:
        test_ids = test_ids[:a.limit]
    print(f"fold {F}: {len(test_ids)} test patients", flush=True)

    for pid in test_ids:
        ip = f"{IMG_DIR}/{pid}_0000.nii.gz"
        if not os.path.exists(ip):
            print(f"MISSING {pid}"); continue
        ref = nib.load(ip); rs = ref.shape; raff = ref.affine
        sp = ref.header.get_zooms()[:3]; img = ref.get_fdata().astype(np.float32)
        zf = [sp[i] / TARGET_ISO for i in range(3)]
        iso = norm(ndzoom(img, zf, order=1))
        vol = iso.transpose(2, 1, 0); D, H, W = vol.shape
        pred = np.zeros((D, H, W), np.uint8)
        with torch.no_grad():
            for z in range(D):
                sl = vol[z]; x, y = sl.shape
                slr = ndzoom(sl, (IMG_SIZE/x, IMG_SIZE/y), order=3) if (x != IMG_SIZE or y != IMG_SIZE) else sl
                inp = torch.from_numpy(slr).unsqueeze(0).unsqueeze(0).float().cuda()
                inp = repeat(inp, 'b c h w -> b (r c) h w', r=3)
                out = net(inp, multimask_output=False, image_size=IMG_SIZE)
                # 2-channel [bg, fg]; argmax over softmax gives class index (SAMed-identical)
                m = torch.argmax(torch.softmax(out['masks'], dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)
                if m.shape[0] != x or m.shape[1] != y:
                    m = ndzoom(m, (x/m.shape[0], y/m.shape[1]), order=0)
                pred[z] = m
        pred_xyz = pred.transpose(2, 1, 0)
        zfb = [rs[i] / pred_xyz.shape[i] for i in range(3)]
        po = ndzoom(pred_xyz, zfb, order=0).astype(np.uint8)
        if po.shape != tuple(rs):
            fx = np.zeros(rs, np.uint8)
            s = tuple(slice(0, min(po.shape[i], rs[i])) for i in range(3)); fx[s] = po[s]; po = fx
        nib.save(nib.Nifti1Image(po, raff), f"{a.out}/{pid}.nii.gz")
        print(f"{pid}: fg voxels={po.sum()}", flush=True)
    print(f"DONE fold {F}")

if __name__ == "__main__":
    main()
