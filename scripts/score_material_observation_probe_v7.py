"""Post-seal full24 task scores and three-layer coverage on ONE fresh TRAIN."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.textured_material_dataset_v1 import cast,transform,inverse
from inspection_decisions.material_observation_inputs_v7 import load_epochs
from inspection_decisions.material_native_matching_frontend_v6 import selected_views
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges,material_tracks
from inspection_decisions.textured_material_matching_v1 import world_points
from scripts.run_material_observation_registration_v7 import METHODS
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_joint_descriptor_comparison_v1 import aggregate
from scripts.score_material_native_modes_v6 import physical_error
from scripts.run_textured_material_comparison_v1 import save,sha


def physically_seen(q,asset,e,frames,truth):
    world=transform(q,np.array(truth['poses'])[e,asset]);seen=np.zeros(len(q),bool)
    for f in frames:
        pc=inverse(world,f['C']);z=pc[:,2];uv=pc[:,:2]/np.maximum(z[:,None],1e-9)*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2];h,w=f['depth'].shape;inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1);ii=np.flatnonzero(inside&~seen)
        if len(ii):
            hit=cast(truth,e,f['C'],uv[ii],f['K']);seen[ii]=(hit['instance']==asset)&(abs(hit['depth_exact']-z[ii])<2e-5)
    return seen


def oracle(truth,anchors,pool,method,fits):
    perm=np.array(truth['permutations']);poses=np.array(truth['poses']);family=0 if method=='base_pool' else 1 if method=='native_ransac' else 2;lookup={tuple(k):h for h,k in enumerate(fits['triples'])};out=[]
    for i,q in enumerate(anchors):
        j,k=perm[1:,i];x=transform(q,poses[0,i]);y1=transform(q,poses[1,i]);y2=transform(q,poses[2,i]);n1,n2=pool['counts'][0,i,j],pool['counts'][1,i,k];A=pool['transforms'][0,i,j,:n1][pool['family_masks'][family,0,i,j,:n1]];B=pool['transforms'][1,i,k,:n2][pool['family_masks'][family,1,i,k,:n2]]
        z1=np.einsum('bij,nj->bni',A[:,:3,:3],x)+A[:,None,:3,3];z2=np.einsum('bij,nj->bni',B[:,:3,:3],x)+B[:,None,:3,3];a=np.quantile(np.linalg.norm(z1-y1,axis=2),.95,axis=1);b=np.quantile(np.linalg.norm(z2-y2,axis=2),.95,axis=1);c=np.quantile(np.linalg.norm((z2[None]-z1[:,None])-(y2-y1),axis=3),.95,axis=2);best=float(np.maximum(np.maximum(a[:,None],b[None]),c).min());h=lookup.get((i,int(j),int(k)))
        if method=='partial_joint' and h is not None:
            for n in range(int(fits[f'{h}_count'])):best=min(best,float(physical_error(*fits[f'{h}_{n}_T'],x,y1,y2)))
        out.append(best)
    return out


def main():
    cfg=json.loads(Path('configs/material_observation_probe_v7.json').read_text(encoding='utf-8'));root=Path(cfg['output']);start=time.monotonic()
    while not (root/'prediction_seal.json').exists():
        if time.monotonic()-start>5400:raise TimeoutError('No complete prediction seal')
        time.sleep(2)
    proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));ncfg=proto['native_config'];rcfg=proto['registration_config'];bcfg=json.loads(Path(cfg['legacy_config']).read_text(encoding='utf-8'));checks=0
    for folder,file in [(root,'prediction_seal.json'),(Path(cfg['native_root']),'correspondence_seal.json')]:
        seal=json.loads((folder/file).read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks+=1
        for p,h in seal['sources'].items():assert sha(p)==h;checks+=1
        for p,h in seal['files'].items():assert sha(folder/p)==h;checks+=1
    dest=Path(cfg['dataset'])/cfg['scene'];release=json.loads((dest.parent/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(dest.parent/p)==h;checks+=1
    # Private scene and material coordinates FIRST enter the evaluator here.
    truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'));truth['poses']=np.array(truth['poses']);truth['geometry']=[np.array(x) for x in truth['geometry']];truth['occluders']=np.array(truth['occluders']);perm=np.array(truth['permutations']);anchors=material_anchors(dest);records=json.loads((dest/'frames.json').read_text(encoding='utf-8'));records={r['index']:r for r in records};labels={};rows=[];coverage=[];physical_cache={}
    for condition in cfg['conditions']:
        for v in cfg['variants']:
            folder=root/condition;frames,ids=load_epochs(cfg,condition,v)
            with np.load(Path(cfg['native_root'])/condition/f'{v}_features.npz') as f:ff=dict(f)
            material={};points={};members={};crop={};contamination=0
            for prefix in [k[:-6] for k in ff if k.endswith('_frame')]:
                e,slot,rank=map(int,prefix.split('_'));fi=int(ff[prefix+'_frame']);f=frames[e][fi];pix=ff[prefix+'_pixel'];raw=ff[prefix+'_raw_xyz'];node=ff[prefix+'_node'];q=ff[prefix+'_quality'];rid=ids[e][fi]
                assert np.max(abs(world_points(f,pix)-raw),initial=0)<1e-12;checks+=1
                if rid not in labels:
                    with np.load(dest/records[rid]['labels']) as f:labels[rid]={'owner':f['instance'],'local':f['material_xyz']}
                local=labels[rid]['local'][pix[:,1],pix[:,0]];owner=labels[rid]['owner'][pix[:,1],pix[:,0]];expected=int(np.flatnonzero(perm[e]==slot)[0]);contamination+=int((owner!=expected).sum())
                for n,point,quality,own,z in zip(node,local,q,owner,ff[prefix+'_xyz']):members.setdefault((e,slot,int(n)),[]).append((float(quality),point,int(own),z))
                a=ff[prefix+'_crop'];crop.setdefault((e,slot),[]).append(int(max(a[2]-a[0],a[3]-a[1])))
            for key,mm in members.items():
                representative=max(mm,key=lambda x:x[0]);material[key]=(representative[1],representative[2]);points[key]=representative[3]
            with np.load(Path(cfg['native_root'])/condition/f'{v}_edges.npz') as f:edges=unpack_edges(dict(f),ncfg)
            tracks,_=material_tracks(edges,ncfg)
            with np.load(folder/f'{v}_pools.npz') as f:pool=dict(f)
            with np.load(folder/f'{v}_partial_fits.npz') as f:fits=dict(f)
            with np.load(folder/f'{v}_cache.npz') as f:cache=dict(f)
            with np.load(folder/f'{v}_search_graph.npz') as f:graph=dict(f)
            lookup={tuple(t):h for h,t in enumerate(cache['triples'])}
            for state,value in zip(graph['states'],graph['values']):
                assert all(np.array_equal(np.sort(state[:,e]),np.arange(24)) for e in range(2));rebuilt=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(state)),start=np.zeros(4));assert np.max(abs(rebuilt-value))<1e-9;checks+=6
            for h in range(len(fits['triples'])):
                for n in range(int(fits[f'{h}_count'])):assert np.max(np.diff(fits[f'{h}_{n}_path']),initial=0)<=1e-12;checks+=1
            for col,m in enumerate(METHODS):
                with np.load(folder/f'{v}_{m}.npz') as f:p=dict(f)
                state=int(p['search_state']);assert state==int(graph['values'][:,col].argmin()) and np.array_equal(p['target_slots'],graph['states'][state]);checks+=1
                for i,(j,k) in enumerate(p['target_slots']):
                    h=lookup[i,int(j),int(k)];assert np.max(abs(p['transforms'][i]-cache['transforms'][h,col]))<1e-12 and abs(p['objectives'][i]-cache['costs'][h,col])<1e-12;checks+=1
                rr=task_rows(cfg['scene'],'train',v,m,p,truth,anchors,bcfg);oo=oracle(truth,anchors,pool,m,fits)
                for i,r in enumerate(rr):r.update(condition=condition,oracle_pool_worst_error_m=oo[i],oracle_pool_qualified=int(oo[i]<=.05));assert r['qualified']<=r['oracle_pool_qualified'];checks+=1
                rows+=rr
            for i,q in enumerate(anchors):
                slots=[i,int(perm[1,i]),int(perm[2,i])];key=condition,i
                if key not in physical_cache:
                    allseen=[physically_seen(q,i,e,frames[e],truth) for e in range(3)];selected=[physically_seen(q,i,e,[frames[e][j] for j in selected_views(frames[e],slots[e],rcfg['score_views_per_slot'])],truth) for e in range(3)];physical_cache[key]=(allseen,selected)
                allseen,selected=physical_cache[key];sampled=[]
                for e in range(3):
                    mm=[value[0] for key,value in material.items() if key[0]==e and key[1]==slots[e] and value[1]==i];d=cKDTree(np.array(mm)).query(q)[0] if mm else np.full(len(q),np.inf);sampled.append(d<=.015)
                track=tracks.get(tuple(slots));td=[];pd=[]
                for pe,(e,f) in enumerate([(0,1),(0,2),(1,2)]):
                    r=edges[pe][slots[e]*24+slots[f]]
                    for a,b in zip(r['source'],r['target']):
                        u,ou=material[e,slots[e],int(a)];w,ow=material[f,slots[f],int(b)];pd.append(float(np.linalg.norm(u-w)) if ou==ow==i else float('inf'))
                if track is not None:
                    for node in track['node']:
                        mm=[material[e,slots[e],int(node[e])] for e in range(3)];td.append(max(float(np.linalg.norm(mm[a][0]-mm[b][0])) for a,b in [(0,1),(0,2),(1,2)]) if all(x[1]==i for x in mm) else float('inf'))
                # A genuinely nested three-layer diagnostic on the SAME native
                # historical source nodes. Fixed anchors above are a separate
                # common-query diagnostic and must not be conflated with cycles.
                source_nodes=[key[2] for key,mm in material.items() if key[0]==0 and key[1]==i and mm[1]==i];source_q=np.array([material[0,i,n][0] for n in source_nodes]).reshape(-1,3);ni={n:h for h,n in enumerate(source_nodes)};physical=[];sample=[];matched=[]
                for pe,e in enumerate([1,2]):
                    vv=[frames[e][j] for j in selected_views(frames[e],slots[e],rcfg['score_views_per_slot'])];physical.append(physically_seen(source_q,i,e,vv,truth));mm=[value[0] for key,value in material.items() if key[0]==e and key[1]==slots[e] and value[1]==i];dist=cKDTree(np.array(mm)).query(source_q)[0] if mm else np.full(len(source_q),np.inf);sample.append(dist<=.015);good=np.zeros(len(source_q),bool);r=edges[pe][i*24+slots[e]]
                    for a,b in zip(r['source'],r['target']):
                        if int(a) not in ni:continue
                        u,ou=material[0,i,int(a)];w,ow=material[e,slots[e],int(b)]
                        if ou==ow==i and np.linalg.norm(u-w)<=.015:good[ni[int(a)]]=True
                    matched.append(good)
                eligible=physical[0]&physical[1];sampled_both=eligible&sample[0]&sample[1];matched_both=sampled_both&matched[0]&matched[1];good_source_cycles=set()
                if track is not None:
                    for node,d in zip(track['node'],td):
                        if d<=.015:good_source_cycles.add(int(node[0]))
                q0,q1,q2=allseen;s0,s1,s2=selected;t0,t1,t2=sampled
                coverage.append(dict(condition=condition,variant=v,asset=i,anchors=len(q),physical_all_h1=int((q0&q1).sum()),physical_all_h2=int((q0&q2).sum()),physical_all_three=int((q0&q1&q2).sum()),physical_selected_three=int((s0&s1&s2).sum()),sampled_three=int((q0&q1&q2&t0&t1&t2).sum()),
                    native_true_edges=len(pd),native_good_edges=sum(d<=.015 for d in pd),native_cycles=len(td),native_good_cycles=sum(d<=.015 for d in td),good_cycle_unique_history_nodes=len(good_source_cycles),native_history_queries=len(source_q),history_queries_physically_visible_both=int(eligible.sum()),history_queries_sampled_both=int(sampled_both.sum()),history_queries_matched_both=int(matched_both.sum()),min_native_points=int(min(sum(1 for key,mm in material.items() if key[0]==e and key[1]==slots[e]) for e in range(3))),median_original_crop_px=float(np.median([x for e in range(3) for x in crop.get((e,slots[e]),[])]))))
            print('V7 SCORED',condition,v,'contaminating native pixels',contamination,flush=True)
    write_csv(root/'task_scores.csv',rows);write_csv(root/'coverage.csv',coverage)
    summaries={}
    for condition in cfg['conditions']:
        summaries[condition]=aggregate([r for r in rows if r['condition']==condition],[],METHODS,cfg['variants'],[dict(baseline='base_pool',candidate='native_ransac'),dict(baseline='native_ransac',candidate='native_graph'),dict(baseline='native_graph',candidate='partial_joint')])
    save(root/'scores.json',summaries);save(root/'scoring_audit.json',dict(complete=True,checks=checks,task_rows=len(rows),source_sha=sha(__file__),all_predictions_sealed_before_private_scoring=True,fixed_common_anchors_across_conditions=True,scene_unit=1,TRAIN_only=True,source_hashes={p:sha(p) for p in ['scripts/score_material_native_modes_v6.py','scripts/score_textured_material_comparison_v1.py']},hashes={p:sha(root/p) for p in ['task_scores.csv','coverage.csv','scores.json']}));print('V7 SCORES COMPLETE',checks,len(rows),flush=True)


if __name__=='__main__':main()
