#!/usr/bin/env python3
"""
eval_complete.py  (v2: faster + optional post-processing)
=========================================================
Evaluate already-saved predictions. No inference, no GPU.

Per patient + aggregate:
  VOXEL:    Dice, Precision, Recall, HD95(mm)
  SURFACE:  Surface Dice @1mm, @2mm, @5mm
  LESION:   detection Recall/Precision (any-overlap AND IoU>=thresh)

POST-PROCESSING (optional, --min_pred_voxels N):
  Removes predicted connected components smaller than N voxels, then recomputes
  the SAME metrics -> columns suffixed _adj. Lets you compare raw vs cleaned.

Saves per_patient.csv + summary.json.
"""
import argparse, json, csv, time
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_erosion

def dice_3d(p,g,s=1e-6):
    i=(p&g).sum(); return float((2*i+s)/(p.sum()+g.sum()+s))
def prec_rec(p,g,s=1e-6):
    tp=(p&g).sum(); fp=(p&~g).sum(); fn=(~p&g).sum()
    return float((tp+s)/(tp+fp+s)), float((tp+s)/(tp+fn+s))

def _bbox(pb,gb,pad=5):
    co=np.argwhere(pb|gb)
    if len(co)==0: return None
    lo=np.maximum(co.min(0)-pad,0); hi=np.minimum(co.max(0)+pad+1,np.array(pb.shape))
    return (slice(lo[0],hi[0]),slice(lo[1],hi[1]),slice(lo[2],hi[2]))

def surf_and_hd(p,g,spacing,tols=(1.,2.,5.)):
    """Compute surface dice (multiple tols) + HD95 in ONE cropped distance-transform pass."""
    if not p.any() or not g.any():
        return {t:0.0 for t in tols}, None
    bb=_bbox(p,g); pb=p[bb]; gb=g[bb]
    pbd=pb&~binary_erosion(pb); gbd=gb&~binary_erosion(gb)
    pdt=distance_transform_edt(~pb,sampling=spacing)
    gdt=distance_transform_edt(~gb,sampling=spacing)
    dp=gdt[pbd]; dg=pdt[gbd]   # distances from each surface to the other
    den=gbd.sum()+pbd.sum()
    sd={t: float(((dg<=t).sum()+(dp<=t).sum())/den) if den>0 else 0.0 for t in tols}
    alld=np.concatenate([dp,dg]) if (len(dp)+len(dg))>0 else np.array([0.])
    hd=float(np.percentile(alld,95))
    return sd, hd

def remove_small(mask, min_vox):
    """Remove connected components < min_vox. Returns cleaned bool mask."""
    if min_vox<=1: return mask
    st=np.ones((3,3,3),int)
    lab,n=ndimage.label(mask,st)
    if n==0: return mask
    sizes=np.bincount(lab.ravel())
    keep=np.where(sizes>=min_vox)[0]; keep=keep[keep>0]
    return np.isin(lab,keep)

def lesion_metrics(p,g,iou_thresh,min_vox):
    """Returns (recall_any, prec_any, recall_iou, prec_iou, n_gt, n_pred)."""
    st=np.ones((3,3,3),int)
    gl,ng=ndimage.label(g,st); pl,npr=ndimage.label(p,st)
    g_sizes=np.bincount(gl.ravel()); p_sizes=np.bincount(pl.ravel())
    gids=set(np.where(g_sizes>=min_vox)[0]) - {0}
    pids=set(np.where(p_sizes>=min_vox)[0]) - {0}
    det_any=set(); mpr_any=set(); det_iou=set(); mpr_iou=set()
    for gi in gids:
        gm=(gl==gi); gsz=g_sizes[gi]
        ov=np.unique(pl[gm]); ov=[int(x) for x in ov if x>0 and x in pids]
        for pi in ov:
            inter=(gm&(pl==pi)).sum()
            if inter>0:
                det_any.add(gi); mpr_any.add(pi)
                union=gsz+p_sizes[pi]-inter
                if union>0 and inter/union>=iou_thresh:
                    det_iou.add(gi); mpr_iou.add(pi)
    ng_e=len(gids); np_e=len(pids)
    ra=len(det_any)/ng_e if ng_e else float('nan')
    pa=len(mpr_any)/np_e if np_e else float('nan')
    ri=len(det_iou)/ng_e if ng_e else float('nan')
    pi_=len(mpr_iou)/np_e if np_e else float('nan')
    return ra,pa,ri,pi_,ng_e,np_e

def all_metrics(pred, gt, spacing, iou_thresh, min_lesion):
    d=dice_3d(pred,gt); pr,rc=prec_rec(pred,gt)
    sd,hd=surf_and_hd(pred,gt,spacing)
    ra,pa,ri,pi_,ng,npr=lesion_metrics(pred,gt,iou_thresh,min_lesion)
    return dict(dice=d,precision=pr,recall=rc,hd95=hd,
                sdice_1mm=sd[1.0],sdice_2mm=sd[2.0],sdice_5mm=sd[5.0],
                les_recall_any=ra,les_prec_any=pa,les_recall_iou=ri,les_prec_iou=pi_,
                n_gt_lesion=ng,n_pred_lesion=npr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_root",required=True); ap.add_argument("--lbl_dir",required=True)
    ap.add_argument("--splits",required=True); ap.add_argument("--out_dir",required=True)
    ap.add_argument("--folds",default="0,1,2,3,4")
    ap.add_argument("--iou_thresh",type=float,default=0.1)
    ap.add_argument("--min_lesion_voxels",type=int,default=100)
    ap.add_argument("--min_pred_voxels",type=int,default=0,
                    help="post-proc: remove predicted components < this; adds _adj metrics")
    a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    folds=[int(x) for x in a.folds.split(",")]
    splits=json.load(open(a.splits)); lbl=Path(a.lbl_dir)

    rows=[]; spacing_note=None
    for fold in folds:
        pdir=Path(a.pred_root)/f"fold_{fold}"
        if not pdir.exists(): print(f"[skip] fold {fold} missing"); continue
        for pid in splits[fold]["val"]:
            pp=pdir/f"{pid}.nii.gz"; gp=lbl/f"{pid}.nii.gz"
            if not pp.exists() or not gp.exists(): print(f"  miss {pid}"); continue
            t0=time.time()
            pn=nib.load(str(pp)); pred=(np.asarray(pn.dataobj)>0)
            gt=(np.asarray(nib.load(str(gp)).dataobj)>0)
            sp=tuple(float(z) for z in pn.header.get_zooms()[:3])
            if spacing_note is None: spacing_note=sp
            m=all_metrics(pred,gt,sp,a.iou_thresh,a.min_lesion_voxels)
            row=dict(fold=fold,pid=pid,**m)
            # post-processing variant
            if a.min_pred_voxels>0:
                pred_adj=remove_small(pred,a.min_pred_voxels)
                madj=all_metrics(pred_adj,gt,sp,a.iou_thresh,a.min_lesion_voxels)
                for k,v in madj.items(): row[f"{k}_adj"]=v
            rows.append(row)
            dt=time.time()-t0
            extra=f" | adjDice={row.get('dice_adj','-')}" if a.min_pred_voxels>0 else ""
            print(f"  f{fold} {pid}: Dice={m['dice']:.3f} Rec={m['recall']:.3f} Prc={m['precision']:.3f} "
                  f"sD2={m['sdice_2mm']:.3f} | lesRec(any)={m['les_recall_any']:.2f} "
                  f"lesRec(iou)={m['les_recall_iou']:.2f} | n_pred={m['n_pred_lesion']} ({dt:.0f}s){extra}",flush=True)

    if not rows: print("No predictions."); return
    keys=list(rows[0].keys())
    with open(out/"per_patient.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    def ms(k):
        v=[r[k] for r in rows if r.get(k) is not None and not (isinstance(r.get(k),float) and np.isnan(r.get(k)))]
        return (float(np.mean(v)),float(np.std(v))) if v else (None,None)
    metric_keys=[k for k in keys if k not in ("fold","pid")]
    summary={k:ms(k) for k in metric_keys}
    print("\n"+"="*60)
    print(f"spacing {tuple(round(s,3) for s in spacing_note)} mm | "+
          " ".join(f"{t}mm={[round(t/s,1) for s in spacing_note]}vox" for t in (1,2,5)))
    print(f"=== POOLED over {len(rows)} patients (mean ± std) ===")
    for k in metric_keys:
        m,s=summary[k]
        if m is not None: print(f"  {k:<22} {m:.4f} ± {s:.4f}")
    json.dump({"spacing":spacing_note,"summary":summary,"n":len(rows),
               "min_lesion_voxels":a.min_lesion_voxels,"min_pred_voxels":a.min_pred_voxels,
               "iou_thresh":a.iou_thresh}, open(out/"summary.json","w"),indent=2)
    print(f"\nSaved: {out}/per_patient.csv + summary.json")

if __name__=="__main__": main()
