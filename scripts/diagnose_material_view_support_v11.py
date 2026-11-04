"""Post-seal all-available versus area-top2 views; diagnostic, not new predictions."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scripts.run_material_inverse_plane_control_v11 import configuration,setup
from inspection_decisions.material_oriented_surface_control_v10 import orient_banks,OrientedScore,inverse
from inspection_decisions.material_inverse_plane_control_v11 import project
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save

def main():
    cfg=configuration();root=Path(cfg['output']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));truth=json.loads((Path(ic['dataset'])/ic['scene']/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);rows=[];frame_rows=[]
    for v in cfg['variants']:
        _,frames,ids,_,_,_,views,banks,_=setup(cfg,v);orient_banks(banks,frames,cfg)
        with np.load(root/f'{v}_raw_patch.npz') as f:pred=dict(f)
        for i in range(24):
            k=int(perm[2,i]);T=poses[2,i]@inverse(poses[0,i]);a,b=banks[0][i],banks[2][k];world=np.einsum('ij,npj->npi',T[:3,:3],a['patch_xyz'])+T[None,None,:3,3];normal=a['patch_normal']@T[:3,:3].T;raw2=np.zeros(world.shape[:-1],bool);rawall=np.zeros_like(raw2);ng2=np.zeros_like(raw2);ngall=np.zeros_like(raw2);selected={id(f) for f in views[2][k]}
            for fi,f in enumerate(frames[2]):
                r=project(world,f,k);hit=r['surface']&(abs(r['gap'])<=cfg['ray_tolerance_m']);cos=np.sum(normal[:,None]*r['normal'],axis=-1);gate=hit&((cos>=cfg['normal_cosine_min'])|~a['patch_normal_valid'][:,None]|~r['normal_valid']);rawall|=hit;ngall|=gate
                if id(f) in selected:raw2|=hit;ng2|=gate
                camera=(world-f['C'][:3,3])@f['C'][:3,:3];z=np.maximum(camera[...,2],1e-8);uv=camera[...,:2]/z[...,None]*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2];span=np.linalg.norm(np.ptp(uv,axis=1),axis=1);good=gate.sum(1)>=3;frame_rows.append(dict(variant=v,asset=i,rear=int(k%3==0),local_frame=fi,public_frame=int(ids[2][fi]),area_top2=int(id(f) in selected),raw_hit_pixels=int(hit.sum()),normal_hit_pixels=int(gate.sum()),supported_patches=int(good.sum()),patch_span_median_px=float(np.median(span[good])) if good.any() else None))
            s2=OrientedScore(a,b,views[0][i],views[2][k],i,k,cfg).score(T[None])[0];sa=OrientedScore(a,b,frames[0],frames[2],i,k,cfg).score(T[None])[0]
            # Same incorrect hypothesis gets the same extra views: avoid a one-sided oracle.
            pk=int(pred['target_slots'][i,1]);U=pred['transforms'][i,1];bad2=OrientedScore(a,banks[2][pk],views[0][i],views[2][pk],i,pk,cfg).score(U[None])[0];bada=OrientedScore(a,banks[2][pk],frames[0],frames[2],i,pk,cfg).score(U[None])[0];rows.append(dict(variant=v,asset=i,rear=int(k%3==0),old_c2_identity_correct=int(pk==k),top2_raw_hit_pixels=int(raw2.sum()),all_raw_hit_pixels=int(rawall.sum()),top2_normal_hit_pixels=int(ng2.sum()),all_normal_hit_pixels=int(ngall.sum()),top2_supported_patches=int((ng2.sum(1)>=3).sum()),all_supported_patches=int((ngall.sum(1)>=3).sum()),true_pair_point_top2=float(s2[0]),true_pair_point_all=float(sa[0]),true_pair_patch_top2=float(s2[1]),true_pair_patch_all=float(sa[1]),wrong_pair_patch_top2=float(bad2[1]),wrong_pair_patch_all=float(bada[1]),true_pair_normal_patch_top2=float(s2[3]),true_pair_normal_patch_all=float(sa[3]),wrong_pair_normal_patch_top2=float(bad2[3]),wrong_pair_normal_patch_all=float(bada[3])))
    write_csv(root/'view_support.csv',rows);write_csv(root/'view_support_per_frame.csv',frame_rows);summary=[]
    for v in cfg['variants']:
        rr=[r for r in rows if r['variant']==v and r['rear']];summary.append(dict(variant=v,rear_top2_hit_pixels=sum(r['top2_normal_hit_pixels'] for r in rr),rear_all_hit_pixels=sum(r['all_normal_hit_pixels'] for r in rr),rear_top2_has3_supported_patches=sum(r['top2_supported_patches']>=3 for r in rr),rear_all_has3_supported_patches=sum(r['all_supported_patches']>=3 for r in rr),normal_patch_true_beats_wrong_top2=sum(r['true_pair_normal_patch_top2']<r['wrong_pair_normal_patch_top2'] for r in rr),normal_patch_true_beats_wrong_all=sum(r['true_pair_normal_patch_all']<r['wrong_pair_normal_patch_all'] for r in rr)))
    save(root/'view_support_summary.json',dict(summary=summary,scope='POST-SEAL diagnostic only. GT transform defines projected historical queries; raw camera/depth/normal measure consistency at20mm, not exact raycaster physical visibility. All views also given to incorrect hypothesis. This is extra view budget and max-over-views selection, not a same-budget algorithm win, not new predictions.'));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
