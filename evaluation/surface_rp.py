import numpy as np, nibabel as nib, glob, os, csv, json
from scipy.ndimage import distance_transform_edt, binary_erosion

def _bbox(p,g):
    m=p|g
    if not m.any(): return tuple(slice(0,1) for _ in range(m.ndim))
    out=[]
    for ax in range(m.ndim):
        idx=np.any(m,axis=tuple(i for i in range(m.ndim) if i!=ax))
        nz=np.where(idx)[0]
        out.append(slice(max(0,nz[0]-1), min(m.shape[ax],nz[-1]+2)))
    return tuple(out)

def surface_rp(p,g,spacing,tols=(1.0,2.0,5.0)):
    if not p.any() or not g.any():
        z={t:0.0 for t in tols}; return z,z
    bb=_bbox(p,g); pb=p[bb]; gb=g[bb]
    pbd=pb&~binary_erosion(pb); gbd=gb&~binary_erosion(gb)
    pdt=distance_transform_edt(~pb,sampling=spacing)
    gdt=distance_transform_edt(~gb,sampling=spacing)
    dp=gdt[pbd]; dg=pdt[gbd]
    srec={t: float((dg<=t).mean()) if dg.size>0 else 0.0 for t in tols}
    sprec={t: float((dp<=t).mean()) if dp.size>0 else 0.0 for t in tols}
    return srec, sprec

def run(pred_root, lbl_dir, splits_path, out_csv, folds):
    splits=json.load(open(splits_path)); rows=[]
    for fold in folds:
        for pid in splits[fold]['val']:
            pp=f"{pred_root}/fold_{fold}/{pid}.nii.gz"; gp=f"{lbl_dir}/{pid}.nii.gz"
            if not (os.path.exists(pp) and os.path.exists(gp)): continue
            pn=nib.load(pp); g=(np.asarray(nib.load(gp).dataobj)>0); p=(np.asarray(pn.dataobj)>0)
            spacing=pn.header.get_zooms()[:3]
            srec,sprec=surface_rp(p,g,spacing)
            row={'pid':pid,'fold':fold}
            for t in (1.0,2.0,5.0):
                row[f'srec_{int(t)}mm']=round(srec[t],4); row[f'sprec_{int(t)}mm']=round(sprec[t],4)
            rows.append(row); print(f"f{fold} {pid}: srec2={srec[2.0]:.3f} sprec2={sprec[2.0]:.3f}")
    if rows:
        with open(out_csv,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        import statistics as st
        print("\n=== SUMMARY (mean over patients) ===")
        for k in [c for c in rows[0] if c.startswith('srec') or c.startswith('sprec')]:
            print(f"  {k}: {st.mean([r[k] for r in rows]):.4f}")
    print(f"\nsaved {out_csv}")

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--pred_root",required=True); a.add_argument("--lbl_dir",required=True)
    a.add_argument("--splits",required=True); a.add_argument("--out_csv",required=True)
    a.add_argument("--folds",default="0,1,2,3,4")
    x=a.parse_args()
    run(x.pred_root,x.lbl_dir,x.splits,x.out_csv,[int(f) for f in x.folds.split(",")])
