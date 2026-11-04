"""Post-seal pose errors, true-energy conflicts and observed surface sidedness."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.material_surface_patch_registration_v9 import bilinear_projection,apply,inverse
from scripts.run_material_surface_patch_registration_v9 import setup,PAIRS,make_all,METHODS
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save

def main():
    cfg=json.loads(Path('configs/material_surface_patch_registration_v9.json').read_text(encoding='utf-8'));root=Path('results/material_surface_patch_registration_v9_run2');assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));truth=json.loads((Path(ic['dataset'])/ic['scene']/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);metrics=list(csv.DictReader((root/'task_metrics.csv').open(encoding='utf-8-sig')));detail=[];normal=[]
    for v in cfg['variants']:
        _,frames,_,bc,ep,groups,views,banks,edges=setup(cfg,v)
        with np.load(root/f'{v}_material_patch.npz') as f:pred=dict(f)
        rr={int(r['asset']):r for r in metrics if r['variant']==v and r['method']=='material_patch'}
        for i in range(24):
            gtT=np.array([poses[e,i]@inverse(poses[0,i]) for e in [1,2]]);est=pred['transforms'][i];R=est[:,:3,:3]@gtT[:,:3,:3].transpose(0,2,1);angles=np.degrees(Rotation.from_matrix(R).magnitude());j,k=map(int,perm[1:,i]);sc=[make_all(pe,a,b,bc,ep,groups,views,banks,edges,cfg) for pe,a,b in [(0,i,j),(1,i,k),(2,j,k)]];gtU=[gtT[0],gtT[1],gtT[1]@inverse(gtT[0])];truecost=sum(w*s.score(T[None])[0,4] for w,s,T in zip([1,1,.5],sc,gtU));detail.append(dict(variant=v,asset=i,rear=rr[i]['rear'],both_ids=rr[i]['both_ids'],qualified=rr[i]['qualified'],worst_error_m=rr[i]['worst_error_m'],rotation_error_h1_deg=float(angles[0]),rotation_error_h2_deg=float(angles[1]),selected_energy=float(pred['objectives'][i]),true_energy=float(truecost),bad_correct_id_beats_true=bool(rr[i]['both_ids']=='1' and rr[i]['qualified']=='0' and pred['objectives'][i]<truecost)))
            if rr[i]['both_ids']!='1':continue
            for kind,T in [('selected',est[1]),('true',gtT[1])]:
                a=banks[0][i];x=a['patch_xyz'];n=np.cross(x[:,2]-x[:,1],x[:,4]-x[:,3]);length=np.linalg.norm(n,axis=1);n/=np.maximum(length[:,None],1e-12);camera=np.array([frames[0][int(fi)]['C'][:3,3] for fi in a['patch_frame']]);n[np.sum(n*(camera-x[:,0]),axis=1)<0]*=-1;ns=np.repeat(n,5,axis=0)@T[:3,:3].T;z=apply(T,x.reshape(-1,3));best=np.full(len(z),np.inf);dot=np.zeros(len(z))
                for f in views[2][k]:
                    r=bilinear_projection(z,f,k,True);hit=r['surface']&(abs(r['gap'])<=cfg['ray_tolerance_m']);take=hit&(abs(r['gap'])<best);best[take]=abs(r['gap'][take]);outward=-r['normal'];outward/=np.maximum(np.linalg.norm(outward,axis=1)[:,None],1e-12);dot[take]=np.sum(ns[take]*outward[take],axis=1)
                hit=np.isfinite(best);valid=np.repeat(length>1e-8,5)&hit;normal.append(dict(variant=v,asset=i,rear=rr[i]['rear'],pose=kind,hit_pixels=int(hit.sum()),valid_normal_pixels=int(valid.sum()),opposite_normal_pixels=int((valid&(dot<0)).sum()),aligned_normal_pixels=int((valid&(dot>.5)).sum()),mean_cosine=float(dot[valid].mean()) if valid.any() else None))
    write_csv(root/'pose_mechanism.csv',detail);write_csv(root/'surface_sidedness.csv',normal);summary=[]
    for v in cfg['variants']:
        q=[r for r in detail if r['variant']==v];back=[r for r in q if r['rear']=='1'];goodid=[r for r in back if r['both_ids']=='1'];summary.append(dict(variant=v,correct_id_rear=len(goodid),rear_h2_rotation_error_over90=sum(r['rotation_error_h2_deg']>90 for r in goodid),bad_correct_id_beats_true=sum(r['bad_correct_id_beats_true'] for r in q),bad_rear_correct_id_beats_true=sum(r['bad_correct_id_beats_true'] for r in goodid)))
    save(root/'pose_mechanism.json',dict(summary=summary,meaning='Post-seal diagnosis. Normal signs use actual source five-pixel geometry and target depth derivatives, not GT normals. GT poses only measure error/energy. Opposite observed normals indicate missing oriented-material compatibility; rotation flips with aligned normals require other evidence. No modified predictions or retuning.'));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
