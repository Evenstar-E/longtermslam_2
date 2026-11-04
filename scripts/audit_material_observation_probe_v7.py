"""Public-input provenance and ordinary joint-fit/selected-energy reconstruction."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges
from inspection_decisions.material_partial_pair_registration_v7 import residual_jacobian
from inspection_decisions.material_correspondence_inference_v3 import inverse
from scripts.run_material_observation_registration_v7 import prepare,make_scorer,METHODS
from scripts.audit_material_native_modes_v6 import native_term
from inspection_decisions.material_joint_id_pose_search_v5 import ordinary_score
from scripts.run_textured_material_comparison_v1 import save,sha


def loss(T,edges):
    r,_,w=residual_jacobian(T,edges);return float(np.sum(w*.025**2*np.log1p(np.sum(r*r,axis=1)/.025**2)))


def main():
    torch.set_num_threads(1);cfg=json.loads(Path('configs/material_observation_probe_v7.json').read_text(encoding='utf-8'));root=Path(cfg['output']);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));rcfg=proto['registration_config'];ncfg=proto['native_config'];native=Path(cfg['native_root']);checks=0;pixels=0;merges=0;maxerr=0.;maxfit=0.;fitcount=0;selectedfit=0
    for folder,file in [(root,'prediction_seal.json'),(native,'correspondence_seal.json')]:
        s=json.loads((folder/file).read_text(encoding='utf-8'));assert s['complete'];checks+=1
        for p,h in s['sources'].items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    for c in cfg['conditions']:
        for v in cfg['variants']:
            frames,ids,bcfg,ep,groups,views=prepare(cfg,rcfg,c,v)
            with np.load(native/c/f'{v}_features.npz') as f:ff=dict(f)
            nodes={};members={}
            for prefix in [k[:-6] for k in ff if k.endswith('_frame')]:
                e,i,k=map(int,prefix.split('_'));fi=int(ff[prefix+'_frame']);f=frames[e][fi];p=ff[prefix+'_pixel'];uv=ff[prefix+'_uv'];crop=ff[prefix+'_crop'];ww,hh=ff[prefix+'_size'];x0,y0,x1,y1=crop;original=(uv-float(ff[prefix+'_origin_offset'])+.5)/[ww/(x1-x0),hh/(y1-y0)]-.5+[x0,y0]
                assert np.max(abs(original-ff[prefix+'_original_uv']),initial=0)<1e-10 and np.array_equal(np.rint(original).astype(int),p);checks+=1
                assert f['depth_valid'][p[:,1],p[:,0]].all() and (f['anonymous_slot'][p[:,1],p[:,0]]==i).all();checks+=1
                assert len(np.unique(p,axis=0))==len(p)<=ncfg['keypoints_per_view'];checks+=1
                assert np.max(abs(old.world_points(f,p)-ff[prefix+'_raw_xyz']),initial=0)<1e-12;checks+=1
                raw=ff[prefix+'_raw_node'];assert np.array_equal(raw,e*100000000+fi*1000000+p[:,1]*f['depth'].shape[1]+p[:,0]);checks+=1
                for n,r,x,q,z in zip(ff[prefix+'_node'],raw,ff[prefix+'_raw_xyz'],ff[prefix+'_quality'],ff[prefix+'_xyz']):members.setdefault((e,i,int(n)),[]).append((int(r),x,float(q),z));nodes[e,i,int(n)]=z
                pixels+=len(p)
            for key,mm in members.items():
                assert len(mm)<=2 and key[-1]==min(x[0] for x in mm);checks+=1
                if len(mm)==2:assert np.linalg.norm(mm[0][1]-mm[1][1])<=ncfg['same_epoch_gate_m']+1e-12;checks+=1;merges+=1
                representative=max(mm,key=lambda x:x[2])[1]
                for _,__,___,x in mm:assert np.max(abs(x-representative))<1e-12;checks+=1
            with np.load(native/c/f'{v}_edges.npz') as f:packed=dict(f)
            for pe,(e,f) in enumerate([(0,1),(0,2),(1,2)]):
                off=packed[f'{pe}_offsets']
                for q in range(576):
                    i,j=divmod(q,24);a,b=off[q:q+2]
                    for n,x in zip(packed[f'{pe}_source'][a:b],packed[f'{pe}_x'][a:b]):assert np.max(abs(nodes[e,i,int(n)]-x))<1e-12;checks+=1
                    for n,x in zip(packed[f'{pe}_target'][a:b],packed[f'{pe}_y'][a:b]):assert np.max(abs(nodes[f,j,int(n)]-x))<1e-12;checks+=1
            edges=unpack_edges(packed,ncfg)
            with np.load(root/c/f'{v}_pools.npz') as f:pool=dict(f)
            with np.load(root/c/f'{v}_partial_fits.npz') as f:fits=dict(f)
            with np.load(root/c/f'{v}_cache.npz') as f:cache=dict(f)
            lookup={tuple(k):h for h,k in enumerate(cache['triples'])};scorers={}
            for h,(i,j,k) in enumerate(fits['triples']):
                rr=[edges[0][i*24+j],edges[1][i*24+k],edges[2][j*24+k]]
                for n in range(int(fits[f'{h}_count'])):
                    path=fits[f'{h}_{n}_path'];err=max(abs(loss(fits[f'{h}_{n}_initial'],rr)-path[0]),abs(loss(fits[f'{h}_{n}_T'],rr)-path[-1]));maxfit=max(maxfit,err);assert err<1e-12 and np.max(np.diff(path),initial=0)<=1e-12;checks+=3;fitcount+=1
            def score(pe,i,j,T):
                key=pe,i,j
                if key not in scorers:scorers[key]=make_scorer(pe,i,j,ep,groups,views,edges,bcfg)
                s=scorers[key];value=ordinary_score(s.ordinary,T[None])[0]+native_term(T,s.edge);assert abs(value-s.score(T[None])[0])<1e-10
                return value
            for m in METHODS:
                col=METHODS.index(m)
                with np.load(root/c/f'{v}_{m}.npz') as f:p=dict(f)
                for i,(j,k) in enumerate(p['target_slots']):
                    j,k=int(j),int(k);h=lookup[i,j,k];T=p['transforms'][i];origin=int(cache['origins'][h]) if m=='partial_joint' else -1
                    if origin>=0:assert np.array_equal(T,fits[f'{h}_{origin}_T']);checks+=1;selectedfit+=1
                    else:
                        family=0 if m=='base_pool' else 1 if m=='native_ransac' else 2
                        for pe,t,G in [(0,j,T[0]),(1,k,T[1])]:
                            n=pool['counts'][pe,i,t];allowed=pool['transforms'][pe,i,t,:n][pool['family_masks'][family,pe,i,t,:n]];assert np.min(np.max(abs(allowed-G),axis=(1,2)))<1e-12;checks+=1
                    R=T[:,:3,:3];assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8 and np.min(np.linalg.det(R))>.999999;checks+=1
                    value=score(0,i,j,T[0])+score(1,i,k,T[1])+.5*score(2,j,k,T[1]@inverse(T[0]));err=abs(value-p['objectives'][i]);maxerr=max(maxerr,float(err));assert err<1e-9;checks+=1
            print('V7 RAW AUDIT',c,v,flush=True)
    save(root/'raw_audit.json',dict(complete=True,checks=checks,source_sha=sha(__file__),native_measured_pixels=pixels,same_epoch_merges=merges,fit_endpoints_rebuilt=fitcount,selected_partial_joint_fits=selectedfit,max_objective_error=maxerr,max_fit_endpoint_error=maxfit,private_labels_read=False,native_forward_independently_repeated=False,all_selected_objectives_reconstructed=True));print('V7 RAW AUDIT COMPLETE',checks,maxerr,flush=True)


if __name__=='__main__':main()
