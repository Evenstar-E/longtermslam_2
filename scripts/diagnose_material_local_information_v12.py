"""Post-seal local derivatives: geometry/texture rank is NOT global observability."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration,setup
from inspection_decisions.material_oriented_surface_control_v10 import orient_banks,inverse
from inspection_decisions.material_inverse_plane_pose_proposals_v12 import project_derivatives
from inspection_decisions.material_correspondence_mode_graph_v6 import skew
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def spectra(A):
    if not len(A):return np.zeros(6),0
    eig=np.maximum(np.linalg.eigvalsh(A.T@A),0)
    return eig,int((eig>max(float(eig[-1])*1e-6,1e-10)).sum())


def main():
    cfg=configuration();root=Path(cfg['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));truth=json.loads((Path(ic['dataset'])/ic['scene']/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);rows=[];raw={}
    for v in cfg['variants']:
        _,frames,_,_,_,_,views,banks,_=setup(cfg,v);orient_banks(banks,frames,cfg)
        for i in range(24):
            a=banks[0][i];slot=int(perm[2,i]);T=poses[2,i]@inverse(poses[0,i]);x=a['patch_xyz'].reshape(-1,3);world=x@T[:3,:3].T+T[:3,3];normal=np.repeat(a['patch_normal']@T[:3,:3].T,5,0);nv=np.repeat(a['patch_normal_valid'],5)
            center=(a['xyz']@T[:3,:3].T+T[:3,3]).mean(0);J=np.concatenate([-skew(world-center),np.broadcast_to(np.eye(3),(len(world),3,3))],2)
            # Centered rotation/translation scaling is declared, not a calibrated prior.
            J*=np.array([.01,.01,.01,.01,.01,.01])[None,None,:]
            choices=[project_derivatives(world,f,slot) for f in views[2][slot]];ok=[]
            for r in choices:ok.append(r['surface']&(abs(r['gap'])<=cfg['ray_tolerance_m'])&((np.sum(normal*r['normal'],1)>=0)|~nv|~r['normal_valid']))
            pick=np.array([np.where(o,abs(r['gap']),np.inf) for r,o in zip(choices,ok)]).argmin(0)
            G=[];F=[];count=0;patches=set();residuals=[]
            for fi,r in enumerate(choices):
                take=ok[fi]&(pick==fi);ix=np.flatnonzero(take);count+=len(ix);patches.update((ix//5).tolist())
                if not len(ix):continue
                G.append(np.einsum('ni,nij->nj',r['gapJ'][ix],J[ix])/.02)
                jac=np.einsum('nci,nij->ncj',r['rgbJ'],J).reshape(-1,5,3,6);color=(r['rgb']-a['patch_rgb'].reshape(-1,3)).reshape(-1,5,3);mask=take.reshape(-1,5);den=np.maximum(mask.sum(1),1)
                jac-=((jac*mask[...,None,None]).sum(1)/den[:,None,None])[:,None];color-=((color*mask[...,None]).sum(1)/den[:,None])[:,None]
                enough=np.repeat(mask.sum(1)>=3,5);ix=np.flatnonzero(take&enough)
                if len(ix):F.append(jac.reshape(-1,3,6)[ix].reshape(-1,6)/cfg['color_scale']);residuals.extend(color.reshape(-1,3)[ix].ravel())
            geom=np.concatenate(G) if G else np.empty((0,6));tex=np.concatenate(F) if F else np.empty((0,6));eg,rg=spectra(geom);ef,rf=spectra(np.r_[geom,tex]);raw[f'{v}_{i}_geometry_eigenvalues']=eg;raw[f'{v}_{i}_fusion_eigenvalues']=ef
            rows.append(dict(variant=v,asset=i,rear=int(slot%3==0),supported_pixels=count,supported_patches=len(patches),centered_texture_rows=len(tex),geometry_local_rank=rg,geometry_texture_local_rank=rf,geometry_min_eigenvalue=float(eg[0]),fusion_min_eigenvalue=float(ef[0]),fusion_max_eigenvalue=float(ef[-1]),true_centered_rgb_rmse=float(np.sqrt(np.mean(np.array(residuals)**2))) if residuals else None))
    write_csv(root/'local_information.csv',rows);np.savez_compressed(root/'local_information_spectra.npz',**raw)
    summary=[]
    for v in cfg['variants']:
        rr=[r for r in rows if r['variant']==v and r['rear']];summary.append(dict(variant=v,rear_with_three_patches_having_any_supported_pixel=sum(r['supported_patches']>=3 for r in rr),rear_with_rank6_geometry=sum(r['geometry_local_rank']==6 for r in rr),rear_with_rank6_geometry_texture=sum(r['geometry_texture_local_rank']==6 for r in rr),rear_no_supported_pixel=[r['asset'] for r in rr if not r['supported_pixels']]))
    save(root/'local_information_summary.json',dict(summary=summary,scope='Private true-pose/true-ID local derivative diagnostic AFTER prediction seal. Fixed H source, forward H2, original top2 views, actual normal-gated pixels only. supported_patches counts blocks with ANY supported pixel, not V11 >=3 pixels per block; do not compare these definitions as an improvement. Relative spectral cutoff1e-6 and centered rot/trans0.01 units. Local numerical rank is not calibrated confidence, global uniqueness or inference success. Native nonzero depth noise can spuriously raise geometry rank.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
