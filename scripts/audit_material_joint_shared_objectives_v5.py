"""Rebuild selected objectives from raw public RGB-D and frozen weights, no GT."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.material_continuous_correspondence_v4 import SurfaceMatcher,surface_grids
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from inspection_decisions.material_joint_id_pose_search_v5 import ordinary_score
from inspection_decisions.material_shared_correspondence_factor_v5 import measured_bank,pixel_footprint,MaterialView,material_energies
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(1);cfg=json.loads(Path('configs/material_joint_shared_v5.json').read_text(encoding='utf-8'));root=Path(cfg['output']);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete']
    base,v4,bcfg=proto['base_config'],proto['v4'],proto['legacy_config'];model=SurfaceMatcher();model.load_state_dict(torch.load(proto['feature_checkpoint'],weights_only=True,map_location='cpu')['state_dict']);model.eval();maxerr=0.;maxfield=0.;checks=0
    for name in proto['scenes']:
        for variant in base['variants']:
            frames=[load_scene_epoch(base['dataset'],name,e,variant,shuffle_seed=base['seed']+e)[0] for e in range(3)]
            lf=[load_scene_epoch(base['dataset'],name,e,variant,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)]
            ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,base) for f in frames]
            grids=[surface_grids(f,g,base,v4) for f,g in zip(frames,groups)];banks=[];queries=[];descs=[]
            with torch.inference_mode():
                for e in range(3):
                    bb=[]
                    for i,g in enumerate(groups[e]):
                        feat=model.encode(torch.from_numpy(g['patch']));bb.append(measured_bank(g,grids[e][i],feat.flatten(2).transpose(1,2).numpy(),cfg))
                        if e==0:
                            q=pixel_footprint(g,frames[0],cfg);queries.append(q);descs.append(model.query(feat).numpy()[q['indices']])
                            # Verify actual pixel/depth bounds without any truth.
                            for h,index in enumerate(q['indices']):
                                f=frames[0][int(g['frame'][index])];delta=(q['nodes'][h]-q['xyz'][h])@f['C'][:3,:3];z=((q['xyz'][h]-f['C'][:3,3])@f['C'][:3,:3])[2]
                                assert abs(delta[:,0]).max()<=z/f['K'][0,0]/2+1e-12 and abs(delta[:,1]).max()<=z/f['K'][1,1]/2+1e-12 and abs(delta[:,2]).max()<=cfg['depth_noise_bound_m']+1e-12;checks+=1
                                pc=(q['nodes'][h]-f['C'][:3,3])@f['C'][:3,:3];image=pc[:,:2]/pc[:,2,None]*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2]
                                assert np.max(abs(image-g['uv'][index]))<=.5+1e-10;checks+=1
                    banks.append(bb)
            scorers={};view_cache={}
            def score(e,f,i,j,T):
                key=e,f,i,j
                if key not in scorers:
                    a,b=ep[e][i],ep[f][j];scorers[key]=PairScore(a,b,old.matches(a,b,bcfg),groups[e][i],groups[f][j],{},frames[f],j,base)
                return ordinary_score(scorers[key],T[None])[0]
            def view(f,i,j,T):
                key=f,i,j,T.tobytes()
                if key not in view_cache:view_cache[key]=MaterialView(queries[i],descs[i],banks[f][j],frames[f],int(j),cfg).likelihood(T[None])[0]
                return view_cache[key]
            with np.load(root/name/f'{variant}_selected_material.npz') as f:evidence=dict(f)
            for m in proto['methods']:
                with np.load(root/name/f'{variant}_{m}.npz') as f:p=dict(f)
                for i,(j,k) in enumerate(p['target_slots']):
                    G1,G2=p['transforms'][i];value=score(0,1,i,int(j),G1)+score(0,2,i,int(k),G2)+.5*score(1,2,int(j),int(k),G2@inverse(G1))
                    if m in ['fusion','independent','shared']:
                        a,b=view(1,i,int(j),G1),view(2,i,int(k),G2);error=max(np.max(abs(a-evidence[f'{m}_{i}_a'])),np.max(abs(b-evidence[f'{m}_{i}_b'])));maxfield=max(maxfield,float(error));assert error<1e-10
                        energy=material_energies(a[None],b[None],queries[i]['node_weights'],queries[i]['area'])[['fusion','independent','shared'].index(m)][0,0]
                        value+=cfg['material_weight']*energy;checks+=2
                    error=abs(value-p['objectives'][i]);maxerr=max(maxerr,float(error));assert error<1e-9,(name,variant,m,i,error);checks+=1
            print('V5 RAW OBJECTIVE VERIFIED',name,variant,flush=True)
    save(root/'objective_reconstruction_audit.json',dict(complete=True,source_sha=sha(__file__),checks=checks,raw_public_data_only=True,
        raw_model_fields_rebuilt=True,max_field_error=maxfield,max_objective_error=maxerr,actual_pixel_bounds_verified=True,
        pixel_prior='Camera-axis diagonal local approximation; actual reprojected pixel/depth bounds checked. Not exact product image-pixel/depth quadrature.'))
    print('V5 RAW OBJECTIVE AUDIT',checks,maxerr,maxfield,flush=True)


if __name__=='__main__':main()
