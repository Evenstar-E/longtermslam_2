"""Post-seal module control: same physical anchors, no 3DMatch rank claims."""
import json
from pathlib import Path
import numpy as np
from scripts.run_pointdsc_taskscale_reference_v15 import configuration,METHODS,load
from scripts.run_material_continuous_surface_correspondence_v14c import setup
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_native_modes_v6 import physical_error
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'] and not seal['private_labels_read'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h,p;checks+=1
    official=Path('third_party/PointDSC_official_v1');manifest=json.loads((official/'source_manifest.json').read_text(encoding='utf-8'))
    for r in manifest['files']:assert sha(official/r['path'])==r['sha256'];checks+=1
    ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene);rows=[];oracles=[]
    for v in c['variants']:
        _,_,_,bc,_,_,_,_,_=setup(c,v);cache=load(root/f'{v}_cache.npz');pool=load(root/f'{v}_proposals.npz');graph=load(root/f'{v}_search_graph.npz');lookup={tuple(t):h for h,t in enumerate(cache['triples'])};old=load(Path(c['parent_root'])/f'{v}_search_graph.npz');assert np.array_equal(graph['states'],old['states']);checks+=1
        for h in range(len(cache['triples'])):
            U=pool[f'{h}_0_transforms'];V=pool[f'{h}_0_values'];R=U[:,:,:3,:3];assert np.isfinite(U).all() and np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<4e-5;assert abs(V.min()-cache['costs'][h,0])<1e-8;checks+=2
        for state,value in zip(graph['states'],graph['values']):
            vv=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(state)),start=np.zeros(1));assert np.max(abs(vv-value))<1e-8;checks+=1
        oracle={}
        for i,q in enumerate(anchors):
            j,k=map(int,perm[1:,i]);h=lookup.get((i,j,k));error=None
            if h is not None:
                x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q);error=float(min(physical_error(*T,x,y1,y2) for T in pool[f'{h}_0_transforms']))
            oracle[i]=error;oracles.append(dict(variant=v,asset=i,rear=int(perm[2,i]%3==0),oracle_error_m=error))
        for col,m in enumerate(METHODS):
            pred=load(root/f'{v}_{m}.npz');state=int(pred['search_state']);assert state==int(graph['best'][col]) and np.array_equal(pred['target_slots'],graph['states'][state]);checks+=1;rr=task_rows(ic['scene'],'train',v,m,pred,truth,anchors,bc)
            for i,(j,k) in enumerate(pred['target_slots']):
                assert np.array_equal(pred['transforms'][i],cache['transforms'][lookup[i,int(j),int(k)],col]);checks+=1;error=oracle[i];rr[i].update(rear=int(perm[2,i]%3==0),oracle_qualified=int(error is not None and error<=.05),oracle_error_m=error);assert rr[i]['qualified']<=rr[i]['oracle_qualified'];checks+=1
            rows+=rr
    write_csv(root/'task_metrics.csv',rows);write_csv(root/'oracle_metrics.csv',oracles);summary=[]
    for v in c['variants']:
        rr=[r for r in rows if r['variant']==v];good=[r for r in rr if r['both_ids']];summary.append(dict(variant=v,method=METHODS[0],qualified=sum(r['qualified'] for r in rr),both_ids=len(good),rear_qualified=sum(r['qualified'] for r in rr if r['rear']),oracle_qualified=sum(r['oracle_qualified'] for r in rr),correct_id_median_error_m=float(np.median([r['worst_error_m'] for r in good])) if good else None))
    scope='Actual official PointDSC algorithm+3DMatch weights CPU, shared V3 full-native features and official-demo top1; preregistered0.05m inlier/NMS and bidirectional0.05m inlier-count task adaptation and fixed V14graph. Not full author FCGF system, same-final-pool ablation, equal correspondence budget or 3DMatch benchmark.'
    save(root/'scores.json',dict(primary_rows=len(rows),checks=checks,summary=summary,scope=scope));sources=list(seal['sources'])+[__file__];files=[p for p in root.iterdir() if p.is_file() and p.name!='completion_integrity.json'];save(root/'completion_integrity.json',dict(complete=True,checks=checks,official_sources_unchanged=True,official_commit=manifest['commit'],source_hashes={p:sha(p) for p in sources},evidence_hashes={str(p):sha(p) for p in files},scope=scope));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
