"""Post-seal task scoring, legal paired-mode oracle, graph/pose audit."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_joint_descriptor_comparison_v1 import aggregate
from scripts.run_material_native_modes_v6 import METHODS,COUPLED
from scripts.run_material_mode_intersection_v6 import ROOT as EXTRA
from scripts.run_textured_material_comparison_v1 import save,sha


def physical_error(G1,G2,x,y1,y2):
    z1=apply(G1,x);z2=apply(G2,x)
    return max(np.quantile(np.linalg.norm(z1-y1,axis=1),.95),np.quantile(np.linalg.norm(z2-y2,axis=1),.95),np.quantile(np.linalg.norm((z2-z1)-(y2-y1),axis=1),.95))


def oracle(truth,pool,anchors,method,modes,extra):
    perm=np.array(truth['permutations']);poses=np.array(truth['poses']);family=0 if method=='old_pool' else 1 if method=='native_ransac' else 2;out=[]
    keyindex={tuple(k):h for h,k in enumerate(modes['keys'])}
    for i,q in enumerate(anchors):
        j,k=perm[1,i],perm[2,i];x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q)
        n1,n2=pool['counts'][0,i,j],pool['counts'][1,i,k];A=pool['transforms'][0,i,j,:n1][pool['family_masks'][family,0,i,j,:n1]];B=pool['transforms'][1,i,k,:n2][pool['family_masks'][family,1,i,k,:n2]]
        z1=np.einsum('bij,nj->bni',A[:,:3,:3],x)+A[:,None,:3,3];z2=np.einsum('bij,nj->bni',B[:,:3,:3],x)+B[:,None,:3,3]
        a=np.quantile(np.linalg.norm(z1-y1,axis=2),.95,axis=1);b=np.quantile(np.linalg.norm(z2-y2,axis=2),.95,axis=1);c=np.quantile(np.linalg.norm((z2[None]-z1[:,None])-(y2-y1),axis=3),.95,axis=2);best=float(np.maximum(np.maximum(a[:,None],b[None]),c).min())
        h=keyindex.get((i,int(j),int(k)))
        if h is not None:
            if method in COUPLED:
                for n in range(int(modes[f'{h}_{method}_count'])):
                    G1,G2=modes[f'{h}_{method}_{n}_T'];best=min(best,float(physical_error(G1,G2,x,y1,y2)))
            if method=='shared_intersection':
                for n in range(int(extra[f'{h}_count'])):
                    G1,G2=extra[f'{h}_{n}_T'];best=min(best,float(physical_error(G1,G2,x,y1,y2)))
        out.append(best)
    return out


def main():
    cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));root=Path(cfg['output']);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));methods=METHODS+['shared_intersection'];checks=0
    for folder in [root,EXTRA]:
        s=json.loads((folder/'prediction_seal.json').read_text(encoding='utf-8'));assert s['complete'] and not s['heldout_labels_read'];checks+=1
        for p,h in s['sources'].items():assert sha(p)==h;checks+=1
        for p,h in s['files'].items():assert sha(folder/p)==h;checks+=1
    native=Path(cfg['native_root']);s=json.loads((native/'correspondence_seal.json').read_text(encoding='utf-8'));assert s['complete'];checks+=1
    for p,h in s['sources'].items():assert sha(p)==h;checks+=1
    for p,h in s['files'].items():assert sha(native/p)==h;checks+=1
    data=Path(proto['base_config']['dataset']);assert sha(data/'release_manifest.json')==proto['dataset_release_sha'];checks+=1
    release=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(data/p)==h;checks+=1
    rows=[];exposure=[]
    for name in proto['scenes']:
        truth=json.loads((data/name/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(data/name);split=json.loads((data/name/'public_metadata.json').read_text(encoding='utf-8'))['split']
        for v in proto['base_config']['variants']:
            with np.load(root/name/f'{v}_pair_pools.npz') as f:pool=dict(f)
            with np.load(Path(cfg['parent_root'])/name/f'{v}_union_pools.npz') as f:parent=dict(f)
            for pe in range(3):
                for i in range(24):
                    for j in range(24):
                        n=int(parent['counts'][pe,i,j]);assert np.array_equal(pool['transforms'][pe,i,j,:n],parent['transforms'][pe,i,j,:n]);assert pool['family_masks'][0,pe,i,j,:n].all();checks+=2
            with np.load(root/name/f'{v}_material_modes.npz') as f:modes=dict(f)
            with np.load(EXTRA/name/f'{v}_intersection_modes.npz') as f:extra=dict(f)
            with np.load(root/name/f'{v}_search_graph.npz') as f:graph=dict(f)
            with np.load(root/name/f'{v}_joint_cache.npz') as f:cache=dict(f)
            lookup={tuple(k):h for h,k in enumerate(cache['triples'])};modeindex={tuple(k):h for h,k in enumerate(modes['keys'])}
            with np.load(EXTRA/name/f'{v}_intersection_cache.npz') as f:ec=dict(f)
            assert np.array_equal(cache['triples'],ec['triples']);checks+=1
            for state,value in zip(graph['states'],graph['values']):
                assert all(np.array_equal(np.sort(state[:,e]),np.arange(24)) for e in [0,1]);checks+=2
                rebuilt=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(state)),start=np.zeros(6));assert np.max(abs(value-rebuilt))<1e-9;checks+=6
            assert np.array_equal(graph['best'],graph['values'].argmin(0));checks+=1
            for h,key in enumerate(modes['keys']):
                for m in COUPLED:
                    for n in range(int(modes[f'{h}_{m}_count'])):assert np.diff(modes[f'{h}_{m}_{n}_path']).max()<=1e-12;checks+=1
                for n in range(int(extra[f'{h}_count'])):assert np.diff(extra[f'{h}_{n}_path']).max()<=1e-12;checks+=1
            for m in methods:
                folder=EXTRA if m=='shared_intersection' else root
                with np.load(folder/name/f'{v}_{m}.npz') as f:p=dict(f)
                state=int(p['search_state']);assert np.array_equal(p['target_slots'],graph['states'][state]);checks+=1
                if m=='shared_intersection':assert state==int(ec['state_values'].argmin());checks+=1
                else:assert state==int(graph['best'][METHODS.index(m)]);checks+=1
                for i,(j,k) in enumerate(p['target_slots']):
                    h=lookup[i,int(j),int(k)];col=METHODS.index(m) if m in METHODS else 0
                    T=ec['transforms'][h] if m=='shared_intersection' else cache['transforms'][h,col];value=ec['costs'][h] if m=='shared_intersection' else cache['costs'][h,col]
                    assert np.max(abs(p['transforms'][i]-T))<1e-10 and abs(p['objectives'][i]-value)<1e-10;checks+=2
                oo=oracle(truth,pool,anchors,m,modes,extra);rr=task_rows(name,split,v,m,p,truth,anchors,proto['legacy_config'])
                for i,r in enumerate(rr):r.update(oracle_pool_worst_error_m=oo[i],oracle_pool_qualified=int(oo[i]<=.05),detail=int(i in truth['detail_assets']))
                rows+=rr
            for i in range(24):
                j,k=np.array(truth['permutations'])[1:,i];h=modeindex.get((i,int(j),int(k)))
                exposure.append(dict(scene=name,variant=v,asset=i,detail=int(i in truth['detail_assets']),tracks=0 if h is None else len(modes[f'{h}_x']),
                    **{m:0 if h is None else int(modes[f'{h}_{m}_count']) for m in COUPLED},shared_intersection=0 if h is None else int(extra[f'{h}_count'])))
            print('V6 AUDITED/SCORED',name,v,flush=True)
    write_csv(root/'task_scores.csv',rows);write_csv(root/'mechanism_exposure.csv',exposure)
    controls=proto['controls']+[dict(baseline='cycle_independent',candidate='shared_intersection'),dict(baseline='joint_all',candidate='shared_intersection'),dict(baseline='shared_modes',candidate='shared_intersection')]
    save(root/'scores.json',aggregate(rows,[],methods,proto['base_config']['variants'],controls))
    save(root/'scoring_audit.json',dict(checks=checks,task_rows=len(rows),source_sha=sha(__file__),all_predictions_sealed_before_truth=True,old_pose_prefixes_byte_identical=True,
        common_search_graph_verified=True,original_and_prescore_intersection_reported=True,hashes={p:sha(root/p) for p in ['scores.json','task_scores.csv','mechanism_exposure.csv']}))
    print('V6 SCORING COMPLETE',checks,len(rows),flush=True)

if __name__=='__main__':main()
