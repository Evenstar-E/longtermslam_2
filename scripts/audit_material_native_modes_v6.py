"""Independent public-RGBD rebuild of native provenance and selected objectives."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from inspection_decisions.material_joint_id_pose_search_v5 import ordinary_score
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges,material_tracks,residual_jacobian
from scripts.run_material_native_modes_v6 import METHODS,COUPLED,NativeScore
from scripts.run_material_mode_intersection_v6 import ROOT as EXTRA
from scripts.run_textured_material_comparison_v1 import save,sha


def native_term(T,r):
    """Scalar dictionary implementation independent of maximum.at batching."""
    a={};b={}
    for x,y,i,j,w in zip(r['x'],r['y'],r['source'],r['target'],r['weight']):
        v=float(w*np.exp(-.5*np.sum((T[:3,:3]@x+T[:3,3]-y)**2)/.025**2))
        a[int(i)]=max(a.get(int(i),0.),v);b[int(j)]=max(b.get(int(j),0.),v)
    return 1-min(1,min(sum(a.values()),sum(b.values()))/6)


def fit_loss(T,x,w):
    r,_=residual_jacobian(T,x);w=w/w.sum();s=.025
    return float(np.sum(w[None]*np.array([1,1,.5])[:,None]*s*s*np.log1p((r*r).sum(2)/(s*s))))


def main():
    torch.set_num_threads(1);cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));root=Path(cfg['output']);native=Path(cfg['native_root']);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));base=proto['base_config'];bcfg=proto['legacy_config']
    for folder,file in [(root,'prediction_seal.json'),(EXTRA,'prediction_seal.json'),(native,'correspondence_seal.json')]:
        seal=json.loads((folder/file).read_text(encoding='utf-8'));assert seal['complete']
        for p,h in seal['sources'].items():assert sha(p)==h
        for p,h in seal['files'].items():assert sha(folder/p)==h
    checks=0;points=0;merges=0;edge_count=0;maxerr=0.;maxnative=0.;fits=0;selected_couples={m:0 for m in COUPLED+['shared_intersection']};selection_duplicate_nodes={m:0 for m in COUPLED+['shared_intersection']};maxfit=0.
    for name in proto['scenes']:
        for v in base['variants']:
            frames=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=base['seed']+e)[0] for e in range(3)]
            lf=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)];ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,base) for f in frames]
            with np.load(native/name/f'{v}_native_features.npz') as f:ff=dict(f)
            nodes={};members={}
            for prefix in [k[:-6] for k in ff if k.endswith('_frame')]:
                e,i,k=map(int,prefix.split('_'));a={key[len(prefix)+1:]:value for key,value in ff.items() if key.startswith(prefix+'_')};frame=frames[e][int(a['frame'])];p=a['pixel'];h,w=frame['depth'].shape
                assert len(p)<=cfg['keypoints_per_view'] and len(np.unique(p,axis=0))==len(p);checks+=1
                assert frame['depth_valid'][p[:,1],p[:,0]].all() and (frame['anonymous_slot'][p[:,1],p[:,0]]==i).all();checks+=1
                assert np.max(abs(old.world_points(frame,p)-a['raw_xyz']),initial=0)<1e-12;checks+=1
                assert np.array_equal(a['raw_node'],e*100000000+int(a['frame'])*1000000+p[:,1]*w+p[:,0]);checks+=1
                x0,y0,x1,y1=a['crop'];ww,hh=a['size'];uv=(a['uv']-float(a['origin_offset'])+.5)/[ww/(x1-x0),hh/(y1-y0)]-.5+[x0,y0]
                assert np.max(abs(uv-a['original_uv']),initial=0)<1e-10 and np.array_equal(np.rint(uv).astype(int),p);checks+=1
                for n,raw,x,q,z in zip(a['node'],a['raw_node'],a['raw_xyz'],a['quality'],a['xyz']):
                    node=int(n);members.setdefault((e,i,node),[]).append((int(raw),x,float(q),z));nodes[e,i,node]=z
                points+=len(p)
            for key,mm in members.items():
                assert len(mm)<=2 and key[-1]==min(x[0] for x in mm);checks+=1
                if len(mm)==2:assert np.linalg.norm(mm[0][1]-mm[1][1])<=cfg['same_epoch_gate_m']+1e-12;merges+=1;checks+=1
                representative=max(mm,key=lambda x:x[2])[1]
                for _,__,___,z in mm:assert np.max(abs(z-representative))<1e-12;checks+=1
            with np.load(native/name/f'{v}_native_edges.npz') as f:packed=dict(f)
            for pe,(e,f) in enumerate([(0,1),(0,2),(1,2)]):
                off=packed[f'{pe}_offsets'];weights=packed[f'{pe}_weight'];assert np.isfinite(weights).all() and (weights>=0).all() and (weights<=1).all();checks+=1
                for q in range(576):
                    i,j=divmod(q,24);a,b=map(int,off[q:q+2])
                    for n,x in zip(packed[f'{pe}_source'][a:b],packed[f'{pe}_x'][a:b]):assert np.max(abs(nodes[e,i,int(n)]-x))<1e-12;checks+=1
                    for n,x in zip(packed[f'{pe}_target'][a:b],packed[f'{pe}_y'][a:b]):assert np.max(abs(nodes[f,j,int(n)]-x))<1e-12;checks+=1
                edge_count+=len(weights)
            edges=unpack_edges(packed,cfg);rebuilt,_=material_tracks(edges,cfg)
            with np.load(root/name/f'{v}_material_modes.npz') as f:mm=dict(f)
            with np.load(EXTRA/name/f'{v}_intersection_modes.npz') as f:xx=dict(f)
            lookup={tuple(k):h for h,k in enumerate(mm['keys'])}
            for h,key in enumerate(mm['keys']):
                tr=rebuilt[tuple(key)]
                for field in ['x','node','weight']:assert np.array_equal(tr[field],mm[f'{h}_{field}']);checks+=1
                for m in COUPLED+['shared_intersection']:
                    source=xx if m=='shared_intersection' else mm;stem=str(h) if m=='shared_intersection' else f'{h}_{m}'
                    for n in range(int(source[f'{stem}_count'])):
                        ix=source[f'{stem}_{n}_selected'];path=source[f'{stem}_{n}_path'];T=source[f'{stem}_{n}_T'];initial=source[f'{stem}_{n}_initial'];x,w=tr['x'][ix],tr['weight'][ix]
                        err=max(abs(fit_loss(initial,x,w)-path[0]),abs(fit_loss(T,x,w)-path[-1]));maxfit=max(maxfit,err);assert err<1e-12 and np.max(np.diff(path))<=1e-12;checks+=3;fits+=1
                        selection_duplicate_nodes[m]+=int(any(len(np.unique(tr['node'][ix,e]))<len(ix) for e in range(3)))
            with np.load(root/name/f'{v}_pair_pools.npz') as f:pool=dict(f)
            with np.load(root/name/f'{v}_joint_cache.npz') as f:cache=dict(f)
            with np.load(EXTRA/name/f'{v}_intersection_cache.npz') as f:extra=dict(f)
            tripleindex={tuple(k):h for h,k in enumerate(cache['triples'])};scorers={}
            def score(pe,i,j,T):
                e,f=[(0,1),(0,2),(1,2)][pe];key=pe,i,j
                if key not in scorers:
                    a,b=ep[e][i],ep[f][j];ordinary=PairScore(a,b,old.matches(a,b,bcfg),groups[e][i],groups[f][j],{},frames[f],j,base);scorers[key]=NativeScore(ordinary,edges[pe][i*24+j])
                s=scorers[key];scalar=ordinary_score(s.ordinary,T[None])[0]+native_term(T,s.edge);batched=s.score(T[None])[0]
                nonlocal maxnative,checks
                maxnative=max(maxnative,abs(scalar-batched));assert abs(scalar-batched)<1e-10;checks+=1
                return scalar
            for m in METHODS+['shared_intersection']:
                folder=EXTRA if m=='shared_intersection' else root
                with np.load(folder/name/f'{v}_{m}.npz') as f:p=dict(f)
                for i,(j,k) in enumerate(p['target_slots']):
                    j,k=int(j),int(k);G1,G2=p['transforms'][i];idx=tripleindex[i,j,k];col=METHODS.index(m) if m in METHODS else 0
                    origin=int(extra['origins'][idx] if m=='shared_intersection' else cache['origins'][idx,col])
                    if origin>=0:
                        selected_couples[m]+=1;h=lookup[i,j,k];T=xx[f'{h}_{origin}_T'] if m=='shared_intersection' else mm[f'{h}_{m}_{origin}_T'];assert np.array_equal(T,p['transforms'][i]);checks+=1
                    else:
                        family=0 if m=='old_pool' else 1 if m=='native_ransac' else 2
                        for pe,t,T in [(0,j,G1),(1,k,G2)]:
                            n=int(pool['counts'][pe,i,t]);allowed=pool['transforms'][pe,i,t,:n][pool['family_masks'][family,pe,i,t,:n]];assert np.min(np.max(abs(allowed-T),axis=(1,2)))<1e-12;checks+=1
                    R=p['transforms'][i,:,:3,:3];assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8 and np.min(np.linalg.det(R))>.999999;checks+=1
                    value=score(0,i,j,G1)+score(1,i,k,G2)+.5*score(2,j,k,G2@inverse(G1));err=abs(value-p['objectives'][i]);maxerr=max(maxerr,float(err));assert err<1e-9;checks+=1
            print('V6 PUBLIC RAW AUDIT',name,v,flush=True)
    save(root/'raw_provenance_objective_audit.json',dict(complete=True,checks=checks,source_sha=sha(__file__),actual_measured_feature_pixels=points,same_epoch_merges=merges,raw_native_edges=edge_count,
        mode_fits_rebuilt=fits,max_selected_objective_error=maxerr,max_scalar_batch_native_error=maxnative,max_fit_endpoint_loss_error=maxfit,selected_coupled_modes=selected_couples,
        modes_containing_repeated_observation_nodes=selection_duplicate_nodes,private_labels_read=False,native_network_independently_rerun=False,
        scope='All native raw-pixel backprojections/canonical merges/edges/cycles and fit endpoints; all 2688 selected full objectives rebuilt. Frozen official forward outputs reused.'))
    print('V6 PUBLIC AUDIT COMPLETE',checks,maxerr,flush=True)


if __name__=='__main__':main()
